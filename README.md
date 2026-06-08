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
git clone https://github.com/harishv27/research-copilot.git
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

## Project Demo

<img width="2125" height="1404" alt="Screenshot 2026-06-08 093204" src="https://github.com/user-attachments/assets/fe268350-94e4-4771-84ac-b20381ac8656" />
<img width="2166" height="1421" alt="Screenshot 2026-06-08 093143" src="https://github.com/user-attachments/assets/3ed00d5c-750a-4969-867f-32bcc1bdfb06" />
<img width="2163" height="1374" alt="Screenshot 2026-06-08 093218" src="https://github.com/user-attachments/assets/702abb74-9b9c-4367-a5fd-e5681d2aa34d" />


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
