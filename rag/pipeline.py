"""
rag/pipeline.py
Persistent ChromaDB RAG with HyDE + MMR diversity re-ranking.
Papers are stored across sessions — re-embedding only happens for new papers.
"""
import hashlib
import asyncio
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_openai import ChatOpenAI

from config import (
    OPENAI_API_KEY, EMBED_MODEL, TOP_K_RETRIEVAL,
    FAST_MODEL, CHROMA_PERSIST_DIR, CHROMA_COLLECTION,
)
from tools.academic_apis import Paper

HYDE_SYSTEM = """You are a scientific paper abstract generator.
Given a research question, write a single-paragraph hypothetical paper abstract
(150-200 words) that would PERFECTLY answer the question with specific findings.
Write only the abstract text — no title, no authors, no labels."""


def _paper_id(p: Paper) -> str:
    key = (p.title or "") + (p.source or "")
    return hashlib.md5(key.encode()).hexdigest()[:16]


class RAGPipeline:
    def __init__(self, persist: bool = True):
        self._persist = persist
        self._client = None
        self._collection = None

    def _init_collection(self):
        embed_fn = OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name=EMBED_MODEL,
        )
        if self._persist:
            Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def index_papers(self, papers: list):
        if self._collection is None:
            self._init_collection()
        documents, ids, metadatas = [], [], []
        for p in papers:
            doc = f"{p.title}\n\n{p.abstract or ''}"
            doc_id = _paper_id(p)
            meta = {
                "title":          p.title or "",
                "year":           str(p.year or ""),
                "venue":          p.venue or "",
                "citation_count": str(p.citation_count or 0),
                "source":         p.source or "",
                "pdf_url":        p.pdf_url or "",
                "doi":            p.doi or "",
                "authors":        ", ".join((p.authors or [])[:4]),
                "evidence_weight": str(p.evidence_weight or 1.0),
            }
            documents.append(doc)
            ids.append(doc_id)
            metadatas.append(meta)
        if documents:
            self._collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        if not self._collection or self._collection.count() == 0:
            return []
        n = min(top_k, self._collection.count())
        results = self._collection.query(
            query_texts=[query], n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        output = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": doc, "metadata": meta,
                "relevance_score": round(1 - dist, 3),
            })
        return output

    def _mmr_rerank(self, candidates: list[dict], top_k: int, lambda_: float = 0.6) -> list[dict]:
        """Maximal Marginal Relevance: balance relevance with diversity."""
        if not candidates:
            return []
        selected = []
        remaining = list(candidates)
        while remaining and len(selected) < top_k:
            if not selected:
                best = max(remaining, key=lambda x: x["relevance_score"])
            else:
                def mmr_score(c):
                    rel  = c["relevance_score"]
                    sims = []
                    for s in selected:
                        c_text = c["text"][:200]
                        s_text = s["text"][:200]
                        c_words = set(c_text.lower().split())
                        s_words = set(s_text.lower().split())
                        if c_words | s_words:
                            sim = len(c_words & s_words) / len(c_words | s_words)
                        else:
                            sim = 0.0
                        sims.append(sim)
                    max_sim = max(sims) if sims else 0.0
                    return lambda_ * rel - (1 - lambda_) * max_sim
                best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)
        return selected

    def reset(self):
        self._client = None
        self._collection = None


async def generate_hyde_query(query: str) -> str:
    try:
        llm = ChatOpenAI(model=FAST_MODEL, api_key=OPENAI_API_KEY, temperature=0.3)
        resp = await llm.ainvoke([
            {"role": "system", "content": HYDE_SYSTEM},
            {"role": "user", "content": query},
        ])
        return resp.content.strip()
    except Exception:
        return query


class HyDERAGPipeline(RAGPipeline):
    async def retrieve_with_hyde(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        hyde_query = await generate_hyde_query(query)
        results_orig = self.retrieve(query, top_k=top_k)
        results_hyde = self.retrieve(hyde_query, top_k=top_k)

        seen = set()
        merged = []
        for chunk in results_orig + results_hyde:
            title = chunk["metadata"].get("title", "")
            if title not in seen:
                seen.add(title)
                merged.append(chunk)

        return self._mmr_rerank(merged, top_k)
