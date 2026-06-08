"""
memory/store.py
SQLite persistence for search history + saved papers.
Semantic search over past queries via lightweight keyword index.
"""
import json
import sqlite3
import time
from pathlib import Path
from typing import Optional
from config import MEMORY_DB_PATH, MAX_HISTORY_ITEMS


def _get_conn() -> sqlite3.Connection:
    Path(MEMORY_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(MEMORY_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    return conn


def _init_db(conn: sqlite3.Connection):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS search_history (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            query            TEXT NOT NULL,
            answer_snippet   TEXT,
            paper_count      INTEGER DEFAULT 0,
            consensus_score  INTEGER DEFAULT 50,
            domain           TEXT DEFAULT 'general',
            elapsed_seconds  REAL DEFAULT 0,
            reflection_count INTEGER DEFAULT 0,
            timestamp        INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS saved_papers (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            query          TEXT,
            title          TEXT NOT NULL,
            authors        TEXT,
            year           INTEGER,
            venue          TEXT,
            doi            TEXT,
            pdf_url        TEXT,
            citation_count INTEGER DEFAULT 0,
            stance         TEXT,
            saved_at       INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_history_ts ON search_history(timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_papers_ts  ON saved_papers(saved_at DESC);
    """)
    conn.commit()


# ─── History ─────────────────────────────────────────────────────────────────

def save_search(query: str, result: dict) -> int:
    conn = _get_conn()
    answer = result.get("answer", "")[:300]
    cursor = conn.execute(
        """INSERT INTO search_history
           (query, answer_snippet, paper_count, consensus_score, domain, elapsed_seconds, reflection_count, timestamp)
           VALUES (?,?,?,?,?,?,?,?)""",
        (
            query, answer,
            len(result.get("papers", [])),
            result.get("consensus", {}).get("score", 50),
            result.get("domain", "general"),
            result.get("elapsed_seconds", 0),
            result.get("reflection_count", 0),
            int(time.time()),
        )
    )
    conn.execute(
        f"DELETE FROM search_history WHERE id NOT IN "
        f"(SELECT id FROM search_history ORDER BY timestamp DESC LIMIT {MAX_HISTORY_ITEMS})"
    )
    conn.commit()
    rid = cursor.lastrowid
    conn.close()
    return rid


def get_history(limit: int = 30) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM search_history ORDER BY timestamp DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def search_history(q: str, limit: int = 10) -> list[dict]:
    """Keyword search over past queries."""
    conn = _get_conn()
    words = [w.lower() for w in q.split() if len(w) > 2]
    if not words:
        rows = conn.execute(
            "SELECT * FROM search_history ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
    else:
        like_clause = " OR ".join(["LOWER(query) LIKE ?" for _ in words])
        params = [f"%{w}%" for w in words] + [limit]
        rows = conn.execute(
            f"SELECT * FROM search_history WHERE {like_clause} ORDER BY timestamp DESC LIMIT ?",
            params
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_history_item(item_id: int):
    conn = _get_conn()
    conn.execute("DELETE FROM search_history WHERE id = ?", (item_id,))
    conn.commit()
    conn.close()


def clear_history():
    conn = _get_conn()
    conn.execute("DELETE FROM search_history")
    conn.commit()
    conn.close()


# ─── Saved Papers ─────────────────────────────────────────────────────────────

def save_paper(query: str, paper: dict) -> int:
    conn = _get_conn()
    authors = paper.get("authors", [])
    authors_str = ", ".join(authors[:4]) if isinstance(authors, list) else str(authors)
    cursor = conn.execute(
        """INSERT INTO saved_papers
           (query, title, authors, year, venue, doi, pdf_url, citation_count, stance, saved_at)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            query,
            paper.get("title", ""),
            authors_str,
            paper.get("year"),
            paper.get("venue", ""),
            paper.get("doi", ""),
            paper.get("pdf_url", ""),
            paper.get("citation_count", 0),
            paper.get("stance", ""),
            int(time.time()),
        )
    )
    conn.commit()
    rid = cursor.lastrowid
    conn.close()
    return rid


def get_saved_papers(limit: int = 50) -> list[dict]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM saved_papers ORDER BY saved_at DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_saved_paper(paper_id: int):
    conn = _get_conn()
    conn.execute("DELETE FROM saved_papers WHERE id = ?", (paper_id,))
    conn.commit()
    conn.close()


def get_stats() -> dict:
    conn = _get_conn()
    total_searches = conn.execute("SELECT COUNT(*) FROM search_history").fetchone()[0]
    saved_papers   = conn.execute("SELECT COUNT(*) FROM saved_papers").fetchone()[0]
    avg_score      = conn.execute("SELECT AVG(consensus_score) FROM search_history").fetchone()[0]
    conn.close()
    return {
        "total_searches": total_searches,
        "saved_papers":   saved_papers,
        "avg_consensus":  round(avg_score or 0),
    }
