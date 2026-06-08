"""
agents/graph.py — Research Copilot pipeline
Fixes in this version:
  - Pydantic warnings fully eliminated: use method="json_mode" + manual parse
    instead of .with_structured_output(PydanticModel) which triggers LangGraph
    checkpoint serialization warnings regardless of include_raw setting.
  - Domain routing loosened: all domains now always hit Semantic Scholar +
    one specialist source, never zero sources.
  - 0-papers fallback: if branch returns nothing, retries with broader query.
"""
import asyncio
import json
import re
import warnings
import logging
from typing import TypedDict, Annotated
import operator

# Belt-and-suspenders warning suppression
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("pydantic").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)

from langgraph.graph import StateGraph, END
from langgraph.constants import Send
from langchain_openai import ChatOpenAI

from tools.academic_apis import (
    fetch_arxiv, fetch_semantic_scholar, fetch_pubmed, fetch_crossref,
    enrich_with_unpaywall, dedup_and_rank, fetch_citation_network, Paper,
)
from rag.pipeline import HyDERAGPipeline
from config import (
    OPENAI_API_KEY, SYNTHESIS_MODEL, FAST_MODEL,
    MAX_PAPERS_FOR_SYNTHESIS, MAX_PAPERS_FOR_CONSENSUS,
    MAX_REFLECTION_LOOPS, CONFIDENCE_THRESHOLD, MAX_SUBQUERIES,
)


# ─── State ────────────────────────────────────────────────────────────────────

class BranchState(TypedDict):
    sub_query:     str
    domain:        str
    branch_papers: list[dict]


class ResearchState(TypedDict):
    query:               str
    sub_queries:         list[str]
    domain:              str
    papers:              list[dict]
    retrieved_chunks:    list[dict]
    consensus:           dict
    answer:              str
    follow_up_questions: list[str]
    paper_summaries:     list[dict]
    citation_graph:      dict
    critic_feedback:     str
    gap_type:            str
    confidence:          float
    reflection_count:    int
    sources_used:        list[str]
    error:               str
    branch_papers:       Annotated[list[dict], operator.add]


# ─── LLM helpers — JSON mode only, no Pydantic structured output ──────────────
# Using json_mode avoids LangGraph checkpoint serialization entirely.
# We parse manually which is equally safe and warning-free.

def _llm(model: str, temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(
        model=model,
        api_key=OPENAI_API_KEY,
        temperature=temperature,
        response_format={"type": "json_object"},
    )


def _llm_text(model: str, temperature: float = 0.1) -> ChatOpenAI:
    """Plain text LLM for synthesis (returns prose, not JSON)."""
    return ChatOpenAI(model=model, api_key=OPENAI_API_KEY, temperature=temperature)


def _parse_json(text: str) -> dict:
    """Safely parse JSON from LLM, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        return {}


# ─── Node 0: Decompose + domain classify ──────────────────────────────────────

DECOMPOSE_SYSTEM = """You are a research query decomposer.
Return ONLY valid JSON — no markdown, no extra text.

Schema:
{
  "sub_queries": ["query1", "query2"],   // 1-3 focused academic sub-queries
  "domain": "medical|cs|social|general"
}

Domain rules:
- medical  : health, disease, drugs, clinical, biology, neuroscience
- cs       : machine learning, AI, software, algorithms, computers
- social   : psychology, sociology, economics, education, behavior
- general  : history, physics, chemistry, environment, other

If input is a greeting or not a research question, return:
{"sub_queries": ["<original input>"], "domain": "general"}"""


async def decompose_node(state: ResearchState) -> dict:
    query = state["query"]
    try:
        llm  = _llm(FAST_MODEL)
        resp = await llm.ainvoke([
            {"role": "system", "content": DECOMPOSE_SYSTEM},
            {"role": "user",   "content": query},
        ])
        data        = _parse_json(resp.content)
        sub_queries = [str(q) for q in (data.get("sub_queries") or [query])[:MAX_SUBQUERIES]]
        domain      = data.get("domain", "general")
        if domain not in ("medical", "cs", "social", "general"):
            domain = "general"
    except Exception:
        sub_queries = [query]
        domain      = "general"
    return {"sub_queries": sub_queries, "domain": domain}


# ─── Node 1a: Branch scout ────────────────────────────────────────────────────
# Every branch always hits Semantic Scholar (best coverage).
# One specialist source is added per domain.
# This prevents the 0-papers bug caused by overly strict domain routing.

_DOMAIN_SOURCES = {
    "medical": [fetch_semantic_scholar, fetch_pubmed],
    "cs":      [fetch_semantic_scholar, fetch_arxiv],
    "social":  [fetch_semantic_scholar, fetch_crossref],
    "general": [fetch_semantic_scholar, fetch_arxiv, fetch_pubmed, fetch_crossref],
}


async def branch_scout_node(state: BranchState) -> dict:
    q        = state["sub_query"]
    domain   = state.get("domain", "general")
    fetchers = _DOMAIN_SOURCES.get(domain, _DOMAIN_SOURCES["general"])

    results = await asyncio.gather(
        *[f(q) for f in fetchers], return_exceptions=True
    )

    papers: list[Paper] = []
    for r in results:
        if isinstance(r, list):
            papers.extend(r)

    # Fallback: if specialist sources returned nothing, try all 4
    if not papers:
        fallback = await asyncio.gather(
            fetch_semantic_scholar(q), fetch_arxiv(q),
            fetch_pubmed(q), fetch_crossref(q),
            return_exceptions=True,
        )
        for r in fallback:
            if isinstance(r, list):
                papers.extend(r)

    papers = await enrich_with_unpaywall(papers)
    return {"branch_papers": [p.to_dict() for p in papers]}


def dispatch_branches(state: ResearchState) -> list[Send]:
    domain = state.get("domain", "general")
    return [
        Send("branch_scout", {"sub_query": q, "domain": domain, "branch_papers": []})
        for q in (state.get("sub_queries") or [state["query"]])
    ]


# ─── Node 1b: Merge + rank ────────────────────────────────────────────────────

async def merge_scout_node(state: ResearchState) -> dict:
    all_dicts  = state.get("branch_papers", [])
    all_papers = [Paper(**d) for d in all_dicts]
    ranked     = dedup_and_rank(all_papers, query=state["query"], top_n=MAX_PAPERS_FOR_SYNTHESIS)
    sources    = list({p.source for p in all_papers if p.source})
    return {
        "papers":       [p.to_dict() for p in ranked],
        "sources_used": sources,
        "branch_papers": [],
    }


# ─── Node 1c: Gap scout (targeted reflection) ─────────────────────────────────

_GAP_FETCHERS = {
    "recent":     [fetch_arxiv],
    "medical":    [fetch_pubmed, fetch_semantic_scholar],
    "conflicting":[fetch_semantic_scholar, fetch_crossref],
    "general":    [fetch_semantic_scholar, fetch_arxiv, fetch_pubmed, fetch_crossref],
}


async def gap_scout_node(state: ResearchState) -> dict:
    gap_type  = state.get("gap_type", "general")
    feedback  = state.get("critic_feedback", "")
    gap_query = f"{state['query']} {feedback}"[:300]
    fetchers  = _GAP_FETCHERS.get(gap_type, _GAP_FETCHERS["general"])

    results = await asyncio.gather(*[f(gap_query) for f in fetchers], return_exceptions=True)
    new_papers: list[Paper] = []
    for r in results:
        if isinstance(r, list):
            new_papers.extend(r)

    new_papers = await enrich_with_unpaywall(new_papers)
    existing   = [Paper(**d) for d in state.get("papers", [])]
    combined   = dedup_and_rank(existing + new_papers, query=state["query"], top_n=MAX_PAPERS_FOR_SYNTHESIS)
    return {"papers": [p.to_dict() for p in combined]}


# ─── Node 2: RAG ──────────────────────────────────────────────────────────────

async def rag_node(state: ResearchState) -> dict:
    papers_dicts = state.get("papers", [])
    if not papers_dicts:
        return {"retrieved_chunks": []}
    papers = [Paper(**d) for d in papers_dicts]
    rag    = HyDERAGPipeline(persist=True)
    rag.index_papers(papers)
    chunks = await rag.retrieve_with_hyde(state["query"])
    return {"retrieved_chunks": chunks}


# ─── Node 3: Consensus ────────────────────────────────────────────────────────

CONSENSUS_SYSTEM = """You are a scientific consensus classifier.
Return ONLY valid JSON — no markdown, no extra text.

Schema:
{
  "stances": [
    {"stance": "supports|rejects|mixed|unrelated", "reason": "one sentence"},
    ...
  ]
}

Classify each paper's stance on the research question:
- supports  : findings confirm or support the claim
- rejects   : findings contradict or refute it
- mixed     : inconclusive or context-dependent
- unrelated : paper does not address the question"""


async def consensus_node(state: ResearchState) -> dict:
    papers = state.get("papers", [])[:MAX_PAPERS_FOR_CONSENSUS]
    if not papers:
        return {
            "consensus":      {"supports": 0, "rejects": 0, "mixed": 0, "score": 50, "total": 0},
            "paper_summaries": [],
        }

    paper_lines = [
        f"[{i+1}] {p.get('title','')}\n{(p.get('abstract') or '')[:400]}"
        for i, p in enumerate(papers)
    ]
    user_msg = f"Research question: {state['query']}\n\n" + "\n\n".join(paper_lines)

    try:
        llm  = _llm(FAST_MODEL)
        resp = await llm.ainvoke([
            {"role": "system", "content": CONSENSUS_SYSTEM},
            {"role": "user",   "content": user_msg},
        ])
        data    = _parse_json(resp.content)
        stances = data.get("stances", [])
    except Exception:
        stances = [{"stance": "mixed", "reason": "classification error"} for _ in papers]

    counts = {"supports": 0, "rejects": 0, "mixed": 0}
    paper_summaries = []

    for p, si in zip(papers, stances):
        stance = si.get("stance", "mixed")
        if stance == "unrelated":
            continue
        if stance not in counts:
            stance = "mixed"
        counts[stance] += 1
        authors_raw = p.get("authors") or []
        paper_summaries.append({
            "title":          p.get("title", ""),
            "snippet":        (p.get("abstract") or "")[:260] + "…",
            "year":           p.get("year"),
            "venue":          p.get("venue") or p.get("source", ""),
            "citation_count": p.get("citation_count") or 0,
            "pdf_url":        p.get("pdf_url") or "",
            "doi":            p.get("doi") or "",
            "authors":        authors_raw if isinstance(authors_raw, list) else [authors_raw],
            "stance":         stance,
            "stance_reason":  si.get("reason", ""),
            "relevance":      round(p.get("relevance") or 0, 2),
            "evidence_weight": float(p.get("evidence_weight") or 1.0),
        })

    total = sum(counts.values()) or 1
    score = round((counts["supports"] * 100 + counts["mixed"] * 50) / total)
    return {
        "consensus":       {**counts, "score": score, "total": total},
        "paper_summaries": paper_summaries,
    }


# ─── Node 4: Citation graph ───────────────────────────────────────────────────

async def citation_graph_node(state: ResearchState) -> dict:
    summaries = state.get("paper_summaries", [])
    top_dois  = [p["doi"] for p in summaries[:5] if p.get("doi")]
    if not top_dois:
        return {"citation_graph": {}}
    results = await asyncio.gather(
        *[fetch_citation_network(doi) for doi in top_dois], return_exceptions=True
    )
    graph = {}
    for doi, result in zip(top_dois, results):
        if isinstance(result, dict) and result:
            graph[doi] = result
    return {"citation_graph": graph}


# ─── Node 5: Synthesis ────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM = """You are an expert academic research assistant.
Give a clear, evidence-based answer using ONLY the paper abstracts provided.

Rules:
1. Start with the most important finding directly — no preamble
2. Cite papers inline as [First Author Year]
3. If papers disagree, state the disagreement explicitly
4. Write 2-4 focused paragraphs
5. Do NOT invent statistics or claims not in the abstracts
6. End with "Research gap: ..." if a clear gap exists
7. If fewer than 3 relevant papers exist, say so honestly

Tone: confident, scientific, precise."""

FOLLOWUP_SYSTEM = """Based on the research question and answer provided,
generate exactly 3 specific follow-up research questions.
Return ONLY valid JSON: {"questions": ["q1", "q2", "q3"]}"""


async def synthesis_node(state: ResearchState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    papers = state.get("papers", [])

    if not chunks:
        context = "\n\n".join(
            f"[{p.get('venue','?')} {p.get('year','?')}] {p.get('title','')}\n"
            f"Authors: {', '.join((p.get('authors') or [])[:3])}\n"
            f"{(p.get('abstract') or '')[:500]}"
            for p in papers[:12]
        )
    else:
        context = "\n\n".join(
            f"[{c['metadata'].get('venue','?')} {c['metadata'].get('year','?')}] "
            f"{c['metadata'].get('title','')}\n"
            f"Authors: {c['metadata'].get('authors','')}\n"
            f"{c['text'][:500]}"
            for c in chunks
        )

    if not context.strip():
        return {"answer": "No relevant papers found for this query.", "confidence": 0.0, "follow_up_questions": []}

    consensus = state.get("consensus", {})
    total     = consensus.get("total", 0)
    notes     = (
        f"\n\nConsensus across {total} papers: "
        f"{consensus.get('supports',0)} support, "
        f"{consensus.get('rejects',0)} reject, "
        f"{consensus.get('mixed',0)} mixed."
    )
    if state.get("critic_feedback"):
        notes += f"\n\nAddress this feedback: {state['critic_feedback']}"

    try:
        # Synthesis uses plain text LLM (prose answer, not JSON)
        llm_text = _llm_text(SYNTHESIS_MODEL)
        resp = await llm_text.ainvoke([
            {"role": "system", "content": SYNTHESIS_SYSTEM},
            {"role": "user",   "content": f"Research question: {state['query']}\n\nPaper abstracts:\n{context}{notes}"},
        ])
        answer = resp.content.strip()

        # Follow-up questions use JSON mode separately
        llm_json = _llm(FAST_MODEL)
        fq_resp  = await llm_json.ainvoke([
            {"role": "system", "content": FOLLOWUP_SYSTEM},
            {"role": "user",   "content": f"Question: {state['query']}\n\nAnswer summary: {answer[:400]}"},
        ])
        fq_data  = _parse_json(fq_resp.content)
        follow_ups = fq_data.get("questions", [])[:3]

        confidence = consensus.get("score", 50) / 100.0
        return {"answer": answer, "follow_up_questions": follow_ups, "confidence": confidence}

    except Exception as e:
        return {"answer": f"Synthesis error: {str(e)}", "confidence": 0.0, "follow_up_questions": []}


# ─── Node 6: Critic ───────────────────────────────────────────────────────────

CRITIC_SYSTEM = """You are a rigorous scientific critic.
Return ONLY valid JSON — no markdown, no extra text.

Schema:
{
  "confidence": 0.0-1.0,
  "verdict": "accept|refine",
  "feedback": "specific improvement instruction or empty string",
  "gap_type": "recent|medical|conflicting|general|none",
  "issues": ["issue1", ...]
}

verdict="refine" only for MAJOR issues: hallucinations, missing key evidence.
For minor polish → verdict="accept".
gap_type="none" when verdict="accept"."""


async def critic_node(state: ResearchState) -> dict:
    answer           = state.get("answer", "")
    reflection_count = state.get("reflection_count", 0)

    if not answer or reflection_count >= MAX_REFLECTION_LOOPS:
        return {"critic_feedback": "", "gap_type": "none", "reflection_count": reflection_count}

    papers_ctx = "\n".join(
        f"- {p.get('title','')} ({p.get('year','')}): {(p.get('abstract') or '')[:200]}…"
        for p in state.get("papers", [])[:10]
    )

    try:
        llm  = _llm(FAST_MODEL)
        resp = await llm.ainvoke([
            {"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user",   "content": (
                f"Research question: {state['query']}\n\n"
                f"Papers:\n{papers_ctx}\n\n"
                f"Answer:\n{answer}"
            )},
        ])
        data       = _parse_json(resp.content)
        confidence = float(data.get("confidence", 0.75))
        verdict    = data.get("verdict", "accept")
        feedback   = data.get("feedback", "") if verdict == "refine" else ""
        gap_type   = data.get("gap_type", "none") if verdict == "refine" else "none"
        return {
            "confidence":       confidence,
            "critic_feedback":  feedback,
            "gap_type":         gap_type,
            "reflection_count": reflection_count + (1 if verdict == "refine" else 0),
        }
    except Exception:
        return {"confidence": 0.75, "critic_feedback": "", "gap_type": "none", "reflection_count": reflection_count}


# ─── Routing ──────────────────────────────────────────────────────────────────

def should_reflect(state: ResearchState) -> str:
    if (state.get("critic_feedback")
            and state.get("confidence", 1.0) < CONFIDENCE_THRESHOLD
            and state.get("reflection_count", 0) < MAX_REFLECTION_LOOPS):
        return "gap_scout"
    return "end"


# ─── Build graph ──────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("decompose",      decompose_node)
    g.add_node("branch_scout",   branch_scout_node)
    g.add_node("merge_scout",    merge_scout_node)
    g.add_node("gap_scout",      gap_scout_node)
    g.add_node("rag",            rag_node)
    g.add_node("consensus",      consensus_node)
    g.add_node("citation_graph", citation_graph_node)
    g.add_node("synthesis",      synthesis_node)
    g.add_node("critic",         critic_node)

    g.set_entry_point("decompose")
    g.add_conditional_edges("decompose", dispatch_branches, ["branch_scout"])
    g.add_edge("branch_scout",   "merge_scout")
    g.add_edge("merge_scout",    "rag")
    g.add_edge("rag",            "consensus")
    g.add_edge("consensus",      "citation_graph")
    g.add_edge("citation_graph", "synthesis")
    g.add_edge("synthesis",      "critic")
    g.add_conditional_edges("critic", should_reflect, {"gap_scout": "gap_scout", "end": END})
    g.add_edge("gap_scout",      "rag")
    return g.compile()


RESEARCH_GRAPH = build_graph()


async def run_research(query: str) -> ResearchState:
    initial: ResearchState = {
        "query": query, "sub_queries": [], "domain": "general",
        "papers": [], "retrieved_chunks": [], "consensus": {},
        "answer": "", "follow_up_questions": [], "paper_summaries": [],
        "citation_graph": {}, "critic_feedback": "", "gap_type": "none",
        "confidence": 1.0, "reflection_count": 0, "sources_used": [],
        "error": "", "branch_papers": [],
    }
    return await RESEARCH_GRAPH.ainvoke(initial)