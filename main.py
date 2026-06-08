"""
main.py — FastAPI backend with real SSE streaming via .astream_events()
Fixes:
  - Pydantic serialization warnings suppressed (warnings.filterwarnings)
  - Query validation: rejects greetings / non-research input before hitting the pipeline
  - SSE final_state capture: walks all end events robustly
  - /api/validate endpoint for frontend pre-check
"""
import asyncio
import json
import re
import time
import warnings
import logging
from pathlib import Path

# ── Suppress Pydantic serializer noise from LangChain structured outputs ──────
warnings.filterwarnings(
    "ignore",
    message=".*PydanticSerializationUnexpectedValue.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*serialized value may not be as expected.*",
    category=UserWarning,
)
logging.getLogger("pydantic").setLevel(logging.ERROR)

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agents.graph import run_research, RESEARCH_GRAPH
from memory.store import (
    save_search, get_history, search_history, delete_history_item, clear_history,
    save_paper, get_saved_papers, delete_saved_paper, get_stats,
)

app = FastAPI(title="Research Copilot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

static_path = Path(__file__).parent / "static"
static_path.mkdir(exist_ok=True)
Path("data/chroma_db").mkdir(parents=True, exist_ok=True)
Path("memory").mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


# ─── Query validation ─────────────────────────────────────────────────────────

# Greetings and non-research patterns to reject immediately
_GREETINGS = re.compile(
    r"^\s*(hi|hello|hey|how are you|how r u|what'?s up|sup|good morning|good evening|"
    r"good afternoon|howdy|yo|hiya|greetings|thanks|thank you|ok|okay|bye|goodbye|"
    r"who are you|what are you|are you (an )?ai|test|testing|[0-9]+|[^a-zA-Z]*)\s*[.!?]*\s*$",
    re.IGNORECASE,
)

_MIN_WORDS = 3       # must have at least 3 words
_MIN_ALPHA = 4       # must have at least 4 alphabetic characters


def validate_research_query(query: str) -> tuple[bool, str]:
    """
    Returns (is_valid, rejection_reason).
    valid = True means proceed to pipeline.
    """
    q = query.strip()

    if not q:
        return False, "Please enter a research question."

    if len(q) > 600:
        return False, "Query is too long. Please keep it under 600 characters."

    if len(q) < 8:
        return False, "Query is too short. Please ask a full research question."

    alpha_count = sum(c.isalpha() for c in q)
    if alpha_count < _MIN_ALPHA:
        return False, "Please enter a valid research question using words."

    word_count = len(q.split())
    if word_count < _MIN_WORDS:
        return False, f"Please ask a complete question (at least {_MIN_WORDS} words)."

    if _GREETINGS.match(q):
        return False, (
            "This looks like a greeting rather than a research question. "
            "Try asking something like: \"Does caffeine improve cognitive performance?\""
        )

    return True, ""


# ─── Models ───────────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str


class SavePaperRequest(BaseModel):
    query: str
    paper: dict


# ─── UI ───────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Place index.html in /static/</h1>")


# ─── Validate endpoint (frontend can call before submitting) ──────────────────

@app.get("/api/validate")
async def validate_endpoint(q: str = ""):
    valid, reason = validate_research_query(q)
    return {"valid": valid, "reason": reason}


# ─── Research POST ────────────────────────────────────────────────────────────

@app.post("/api/research")
async def research_endpoint(req: ResearchRequest):
    valid, reason = validate_research_query(req.query)
    if not valid:
        raise HTTPException(status_code=400, detail=reason)

    start = time.time()
    try:
        result  = await run_research(req.query.strip())
        elapsed = round(time.time() - start, 1)
        response = _build_response(req.query.strip(), result, elapsed)
        try:
            save_search(req.query.strip(), response)
        except Exception:
            pass
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _build_response(query: str, result: dict, elapsed: float) -> dict:
    return {
        "ok":                  True,
        "elapsed_seconds":     elapsed,
        "query":               query,
        "sub_queries":         result.get("sub_queries", []),
        "domain":              result.get("domain", "general"),
        "answer":              result.get("answer", ""),
        "follow_up_questions": result.get("follow_up_questions", []),
        "consensus":           result.get("consensus", {}),
        "papers":              result.get("paper_summaries", []),
        "citation_graph":      result.get("citation_graph", {}),
        "total_found":         len(result.get("papers", [])),
        "confidence":          result.get("confidence", 1.0),
        "reflection_count":    result.get("reflection_count", 0),
        "sources_used":        result.get("sources_used", []),
    }


# ─── SSE streaming ────────────────────────────────────────────────────────────

_NODE_LABELS = {
    "decompose":      ("🧠", "Decomposing research question…"),
    "branch_scout":   ("🔍", "Scouting papers across sources…"),
    "merge_scout":    ("🔀", "Merging & ranking results…"),
    "gap_scout":      ("🔁", "Gap-filling with targeted search…"),
    "rag":            ("🧩", "HyDE indexing into vector store…"),
    "consensus":      ("⚖️",  "Classifying paper stances…"),
    "citation_graph": ("🕸️",  "Building citation network…"),
    "synthesis":      ("✍️",  "Synthesizing answer…"),
    "critic":         ("🔬", "Critic agent reviewing quality…"),
}


@app.get("/api/stream")
async def stream_research(query: str = ""):
    valid, reason = validate_research_query(query)
    if not valid:
        # Return a proper SSE error immediately so the frontend handles it cleanly
        async def _err():
            yield f"data: {json.dumps({'type':'validation_error','message':reason})}\n\n"
        return StreamingResponse(_err(), media_type="text/event-stream",
                                 headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})

    async def event_generator():
        start = time.time()
        q = query.strip()
        initial = {
            "query": q, "sub_queries": [], "domain": "general",
            "papers": [], "retrieved_chunks": [], "consensus": {},
            "answer": "", "follow_up_questions": [], "paper_summaries": [],
            "citation_graph": {}, "critic_feedback": "", "gap_type": "none",
            "confidence": 1.0, "reflection_count": 0, "sources_used": [],
            "error": "", "branch_papers": [],
        }

        final_state = None

        try:
            async for event in RESEARCH_GRAPH.astream_events(initial, version="v2"):
                kind = event.get("event", "")
                name = event.get("name", "")
                data = event.get("data", {})

                # ── Stage progress ──
                if kind == "on_chain_start" and name in _NODE_LABELS:
                    icon, msg = _NODE_LABELS[name]
                    yield f"data: {json.dumps({'type':'stage','stage':name,'icon':icon,'message':msg})}\n\n"
                    await asyncio.sleep(0)

                # ── Sub-queries revealed after decompose ──
                elif kind == "on_chain_end" and name == "decompose":
                    out = data.get("output") or {}
                    if isinstance(out, dict):
                        subs   = out.get("sub_queries", [])
                        domain = out.get("domain", "general")
                    else:
                        subs, domain = [], "general"
                    if subs:
                        yield f"data: {json.dumps({'type':'sub_queries','sub_queries':subs,'domain':domain})}\n\n"
                        await asyncio.sleep(0)

                # ── Paper count update after merge ──
                elif kind == "on_chain_end" and name == "merge_scout":
                    out = data.get("output") or {}
                    if isinstance(out, dict):
                        n = len(out.get("papers", []))
                        if n:
                            yield f"data: {json.dumps({'type':'paper_count','count':n})}\n\n"
                            await asyncio.sleep(0)

                # ── Capture final state — try multiple event names ──
                elif kind == "on_chain_end":
                    out = data.get("output")
                    if isinstance(out, dict) and out.get("answer"):
                        # Any node output that already has an answer is a good candidate
                        # Prefer the most complete one (has paper_summaries)
                        if final_state is None or (
                            out.get("paper_summaries") and not final_state.get("paper_summaries")
                        ):
                            final_state = out

        except Exception as e:
            yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"
            return

        # ── Fallback: run synchronously if streaming didn't capture state ──
        if final_state is None or not final_state.get("answer"):
            try:
                final_state = await run_research(q)
            except Exception as e:
                yield f"data: {json.dumps({'type':'error','message':str(e)})}\n\n"
                return

        elapsed  = round(time.time() - start, 1)
        response = _build_response(q, final_state, elapsed)
        try:
            save_search(q, response)
        except Exception:
            pass
        yield f"data: {json.dumps({'type':'done', **response})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ─── Memory endpoints ─────────────────────────────────────────────────────────

@app.get("/api/history")
async def history_endpoint(limit: int = 30):
    return {"ok": True, "history": get_history(limit)}


@app.get("/api/history/search")
async def history_search_endpoint(q: str, limit: int = 10):
    return {"ok": True, "history": search_history(q, limit)}


@app.delete("/api/history/{item_id}")
async def delete_history_endpoint(item_id: int):
    delete_history_item(item_id)
    return {"ok": True}


@app.delete("/api/history")
async def clear_history_endpoint():
    clear_history()
    return {"ok": True}


@app.post("/api/papers/save")
async def save_paper_endpoint(req: SavePaperRequest):
    pid = save_paper(req.query, req.paper)
    return {"ok": True, "id": pid}


@app.get("/api/papers")
async def get_papers_endpoint(limit: int = 50):
    return {"ok": True, "papers": get_saved_papers(limit)}


@app.delete("/api/papers/{paper_id}")
async def delete_paper_endpoint(paper_id: int):
    delete_saved_paper(paper_id)
    return {"ok": True}


@app.get("/api/stats")
async def stats_endpoint():
    return {"ok": True, **get_stats()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)