# 🔬 Research Copilot

> AI-powered academic search engine — ask a research question, get a synthesized answer from peer-reviewed papers, a consensus meter, citation graph, and paper library — driven by a 9-node multi-agent LangGraph pipeline.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-FF6B35?style=flat)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat&logo=openai)](https://openai.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector--Store-E44D26?style=flat)](https://trychroma.com)

---

## What it does

Type any research question → the system searches 4 live academic databases in parallel, ranks papers by evidence quality, builds a citation network, synthesizes a grounded answer with inline citations, and shows you exactly how much scientific consensus exists — all in under 30 seconds.

---

## Pipeline

```
User Query
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│  [decompose_node]  ←  GPT-4o-mini: breaks query into 1–3        │
│                       focused sub-queries + domain classify      │
│                       (medical / cs / social / general)          │
└──────────────────────────────┬──────────────────────────────────┘
                               │  LangGraph Send() — parallel branches
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
     [branch_scout]   [branch_scout]   [branch_scout]
      sub-query 1      sub-query 2      sub-query 3
      (domain-weighted sources per branch)
      ArXiv · Semantic Scholar · PubMed · CrossRef
              │                │                │
              └────────────────┼────────────────┘
                               ▼
              [merge_scout_node]  ←  dedup + evidence-weighted rank
                               │     (study type × citations × recency)
                               ▼
              [rag_node]  ←  HyDE: generate hypothetical abstract
                             embed with text-embedding-3-small
                             retrieve via MMR diversity re-ranking
                             persistent ChromaDB (cross-session)
                               │
                               ▼
              [consensus_node]  ←  GPT-4o-mini: per-paper stance
                                   supports / rejects / mixed
                               │
                               ▼
              [citation_graph_node]  ←  CrossRef: reference network
                                        for top-5 papers
                               │
                               ▼
              [synthesis_node]  ←  GPT-4o: grounded answer
                                   inline citations [Author Year]
                                   + 3 follow-up questions
                               │
                               ▼
              [critic_node]  ←  GPT-4o-mini: confidence score
                                gap_type classification
                               │
              ┌────────────────┴──────────────────┐
              ▼                                   ▼
     confidence ≥ 0.55                   confidence < 0.55
          END                         [gap_scout_node]
                                      targeted re-fetch
                                      (max 2 loops)
                                           │
                                           └──→ rag_node
```

---

## Features

| Feature | Detail |
|---|---|
| **Query decomposition** | Breaks complex questions into 1–3 focused sub-queries |
| **Domain-aware routing** | Medical → PubMed, CS → ArXiv, Social → CrossRef |
| **HyDE + MMR retrieval** | Hypothetical Document Embedding + diversity re-ranking |
| **Evidence weighting** | Meta-analysis > RCT > Cohort > Case report scoring |
| **Consensus meter** | Visual supports / mixed / rejects breakdown with % score |
| **Citation graph** | Reference network for top-5 papers via CrossRef |
| **Critic loop** | Targeted gap-fill reflection (not full re-fetch) |
| **Follow-up questions** | 3 clickable follow-up queries generated per answer |
| **Persistent RAG** | ChromaDB persists across sessions — no re-embedding |
| **Search history** | SQLite with keyword search over past queries |
| **Paper library** | Save papers with PDF links and stance tags |
| **Real-time SSE** | Live pipeline progress via `.astream_events()` |

---

## Tech Stack

| Component | Technology |
|---|---|
| Agent orchestration | **LangGraph** — 9-node DAG, conditional edges, Send() parallelism |
| LLM — synthesis | **GPT-4o** |
| LLM — fast tasks | **GPT-4o-mini** (decompose, consensus, critic, follow-ups) |
| Embeddings | **text-embedding-3-small** (OpenAI) |
| Vector store | **ChromaDB** persistent — HyDE + MMR retrieval |
| Academic APIs | **ArXiv** · **Semantic Scholar** · **PubMed** · **CrossRef** (all free) |
| Open-access PDFs | **Unpaywall** (free, email only) |
| Memory | **SQLite** — search history + saved paper library |
| Backend | **FastAPI** + uvicorn async + SSE streaming |
| Frontend | Vanilla HTML/CSS/JS — Fraunces + DM Sans typography |

---

## Setup

### Requirements
- Python 3.10+
- OpenAI API key

### Install

```bash
git clone https://github.com/yourusername/research-copilot.git
cd research-copilot
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

```env
OPENAI_API_KEY=sk-your-key-here
UNPAYWALL_EMAIL=your@email.com
```

### Run

```bash
python main.py
```

Open → [http://localhost:8000](http://localhost:8000)

---

## Project Structure

```
research_copilot/
├── main.py                  ← FastAPI app + SSE stream + all endpoints
├── config.py                ← All tunable settings
├── requirements.txt
├── .env.example
│
├── agents/
│   └── graph.py             ← 9-node LangGraph pipeline
│
├── rag/
│   └── pipeline.py          ← HyDE + MMR + persistent ChromaDB
│
├── tools/
│   └── academic_apis.py     ← ArXiv, Semantic Scholar, PubMed, CrossRef, Unpaywall
│
├── memory/
│   └── store.py             ← SQLite history + saved papers
│
├── data/
│   └── chroma_db/           ← Persistent vector store (auto-created)
│
└── static/
    └── index.html           ← Full frontend (single file)
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serve frontend UI |
| `POST` | `/api/research` | Run full pipeline, returns JSON |
| `GET` | `/api/stream?query=...` | SSE live pipeline events |
| `GET` | `/api/validate?q=...` | Pre-validate query before search |
| `GET` | `/api/history` | Get search history |
| `GET` | `/api/history/search?q=...` | Keyword search over history |
| `DELETE` | `/api/history/{id}` | Delete one history item |
| `DELETE` | `/api/history` | Clear all history |
| `POST` | `/api/papers/save` | Save paper to library |
| `GET` | `/api/papers` | Get saved papers |
| `DELETE` | `/api/papers/{id}` | Remove saved paper |
| `GET` | `/api/stats` | Usage statistics |

---

## Speed Tuning

Edit `config.py` to reduce response time:

```python
MAX_ARXIV_RESULTS        = 6    # default 12
MAX_SEMANTIC_RESULTS     = 6    # default 12
MAX_PAPERS_FOR_SYNTHESIS = 8    # default 14
MAX_PAPERS_FOR_CONSENSUS = 8    # default 16
MAX_REFLECTION_LOOPS     = 1    # default 2
MAX_SUBQUERIES           = 2    # default 3
SOURCE_TIMEOUT           = 6.0  # default 9.0
```

---

## License

MIT