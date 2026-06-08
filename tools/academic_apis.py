"""
tools/academic_apis.py
Async wrappers for ArXiv, Semantic Scholar, PubMed, CrossRef + Unpaywall enrichment.
Each source has an individual asyncio.wait_for timeout so one slow API never blocks others.
Relevance scoring uses keyword overlap + evidence-type weighting.
"""
import asyncio
import math
import re
from typing import Optional

import arxiv
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    MAX_ARXIV_RESULTS, MAX_SEMANTIC_RESULTS,
    MAX_PUBMED_RESULTS, MAX_CROSSREF_RESULTS,
    UNPAYWALL_EMAIL, MIN_KEYWORD_OVERLAP, SOURCE_TIMEOUT,
)

_STOPWORDS = {
    "a","an","the","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "can","of","in","on","at","to","for","with","by","from","about","into",
    "through","during","and","or","but","if","as","it","its","this","that",
    "these","those","what","which","who","how","when","where","whether",
    "there","their","they","we","our","you","your","i","my","me","us",
}

_STUDY_TYPE_WEIGHTS = {
    "meta-analysis": 1.4, "systematic review": 1.3, "randomized controlled": 1.2,
    "randomised controlled": 1.2, "rct": 1.2, "cohort study": 1.0,
    "case-control": 0.9, "observational": 0.85, "case report": 0.6,
    "review": 0.8, "survey": 0.75,
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return {w for w in words if w not in _STOPWORDS}


def _relevance_score(title: str, abstract: str, query_keywords: set[str]) -> float:
    if not query_keywords:
        return 1.0
    hay = _keywords(title + " " + abstract)
    return len(query_keywords & hay) / len(query_keywords)


def _evidence_weight(abstract: str) -> float:
    ab_lower = abstract.lower()
    for phrase, weight in _STUDY_TYPE_WEIGHTS.items():
        if phrase in ab_lower:
            return weight
    return 1.0


def relevance_filter(papers: list, query: str) -> list:
    qk = _keywords(query)
    if not qk:
        return papers
    return [p for p in papers if _relevance_score(p.title or "", p.abstract or "", qk) >= MIN_KEYWORD_OVERLAP]


class Paper:
    __slots__ = (
        "title", "abstract", "authors", "year", "doi",
        "venue", "citation_count", "pdf_url", "source",
        "paper_id", "relevance", "evidence_weight",
    )

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.__slots__:
                setattr(self, k, v)
        for slot in self.__slots__:
            if not hasattr(self, slot):
                setattr(self, slot, None)
        if self.relevance is None:
            self.relevance = 0.0
        if self.evidence_weight is None:
            self.evidence_weight = 1.0

    def to_dict(self):
        return {s: getattr(self, s) for s in self.__slots__}


async def _safe_fetch(coro, source_name: str) -> list:
    try:
        return await asyncio.wait_for(coro, timeout=SOURCE_TIMEOUT)
    except asyncio.TimeoutError:
        return []
    except Exception:
        return []


def _build_arxiv_query(query: str) -> str:
    escaped = query.replace('"', '')
    return f'all:"{escaped}"'


async def fetch_arxiv(query: str) -> list:
    loop = asyncio.get_event_loop()
    qk = _keywords(query)

    def _search():
        client = arxiv.Client()
        search = arxiv.Search(
            query=_build_arxiv_query(query),
            max_results=MAX_ARXIV_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for r in client.results(search):
            ab = r.summary.replace("\n", " ")
            p = Paper(
                title=r.title,
                abstract=ab,
                authors=[a.name for a in r.authors],
                year=r.published.year if r.published else None,
                doi=r.doi,
                venue="ArXiv",
                citation_count=0,
                pdf_url=r.pdf_url,
                source="arxiv",
                paper_id=r.entry_id,
                relevance=_relevance_score(r.title, ab, qk),
                evidence_weight=_evidence_weight(ab),
            )
            papers.append(p)
        return papers

    raw = await _safe_fetch(loop.run_in_executor(None, _search), "arxiv")
    return relevance_filter(raw, query)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8))
async def _fetch_semantic_scholar_raw(query: str) -> list:
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": MAX_SEMANTIC_RESULTS,
        "fields": (
            "title,abstract,authors,year,externalIds,"
            "venue,citationCount,openAccessPdf,referenceCount,"
            "influentialCitationCount,publicationTypes"
        ),
    }
    qk = _keywords(query)
    async with httpx.AsyncClient(timeout=12) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
    papers = []
    for item in data.get("data", []):
        abstract = item.get("abstract") or ""
        if not abstract:
            continue
        doi = (item.get("externalIds") or {}).get("DOI")
        oa = item.get("openAccessPdf")
        pdf_url = oa.get("url") if oa else None
        p = Paper(
            title=item.get("title", ""),
            abstract=abstract.replace("\n", " "),
            authors=[a["name"] for a in (item.get("authors") or [])],
            year=item.get("year"),
            doi=doi,
            venue=item.get("venue") or "Semantic Scholar",
            citation_count=item.get("citationCount") or 0,
            pdf_url=pdf_url,
            source="semantic_scholar",
            paper_id=item.get("paperId"),
            relevance=_relevance_score(item.get("title", ""), abstract, qk),
            evidence_weight=_evidence_weight(abstract),
        )
        papers.append(p)
    return papers


async def fetch_semantic_scholar(query: str) -> list:
    raw = await _safe_fetch(_fetch_semantic_scholar_raw(query), "semantic_scholar")
    return relevance_filter(raw, query)


async def _fetch_pubmed_raw(query: str) -> list:
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    fetch_url  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    qk = _keywords(query)
    async with httpx.AsyncClient(timeout=12) as client:
        sr = await client.get(search_url, params={
            "db": "pubmed", "term": query,
            "retmax": MAX_PUBMED_RESULTS, "retmode": "json", "sort": "relevance",
        })
        sr.raise_for_status()
        ids = sr.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        fr = await client.get(fetch_url, params={
            "db": "pubmed", "id": ",".join(ids),
            "retmode": "xml", "rettype": "abstract",
        })
        fr.raise_for_status()
        xml = fr.text

    papers = []
    articles = re.findall(r"<PubmedArticle>(.*?)</PubmedArticle>", xml, re.DOTALL)
    for art in articles:
        title_m    = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", art, re.DOTALL)
        abstract_m = re.search(r"<AbstractText.*?>(.*?)</AbstractText>", art, re.DOTALL)
        year_m     = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", art, re.DOTALL)
        pmid_m     = re.search(r"<PMID.*?>(\d+)</PMID>", art)
        title    = re.sub(r"<[^>]+>", "", title_m.group(1) if title_m else "").strip()
        abstract = re.sub(r"<[^>]+>", "", abstract_m.group(1) if abstract_m else "").strip()
        year     = int(year_m.group(1)) if year_m else None
        pmid     = pmid_m.group(1) if pmid_m else ""
        author_blocks = re.findall(r"<Author[^>]*>(.*?)</Author>", art, re.DOTALL)
        authors = []
        for ab in author_blocks:
            ln = re.search(r"<LastName>(.*?)</LastName>", ab)
            fn = re.search(r"<ForeName>(.*?)</ForeName>", ab)
            if ln:
                authors.append((fn.group(1) + " " if fn else "") + ln.group(1))
        if not title or not abstract:
            continue
        p = Paper(
            title=title, abstract=abstract, authors=authors, year=year,
            doi=None, venue="PubMed", citation_count=0,
            pdf_url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else None,
            source="pubmed", paper_id=f"pubmed_{pmid}",
            relevance=_relevance_score(title, abstract, qk),
            evidence_weight=_evidence_weight(abstract),
        )
        papers.append(p)
    return papers


async def fetch_pubmed(query: str) -> list:
    raw = await _safe_fetch(_fetch_pubmed_raw(query), "pubmed")
    return relevance_filter(raw, query)


async def _fetch_crossref_raw(query: str) -> list:
    url = "https://api.crossref.org/works"
    qk = _keywords(query)
    async with httpx.AsyncClient(timeout=12) as client:
        resp = await client.get(url, params={
            "query": query, "rows": MAX_CROSSREF_RESULTS,
            "select": "title,abstract,author,published,DOI,container-title,is-referenced-by-count",
            "mailto": UNPAYWALL_EMAIL,
        })
        resp.raise_for_status()
        items = resp.json().get("message", {}).get("items", [])
    papers = []
    for item in items:
        titles = item.get("title", [])
        title  = titles[0] if titles else ""
        abstract = re.sub(r"<[^>]+>", "", item.get("abstract", "")).strip()
        if not title or not abstract:
            continue
        year = None
        pd = item.get("published", {}).get("date-parts", [[]])
        if pd and pd[0]:
            year = pd[0][0]
        authors = []
        for a in (item.get("author") or []):
            name = (a.get("given", "") + " " + a.get("family", "")).strip()
            if name:
                authors.append(name)
        doi   = item.get("DOI", "")
        venue = (item.get("container-title") or ["CrossRef"])[0]
        p = Paper(
            title=title, abstract=abstract, authors=authors, year=year,
            doi=doi, venue=venue,
            citation_count=item.get("is-referenced-by-count", 0),
            pdf_url=f"https://doi.org/{doi}" if doi else None,
            source="crossref", paper_id=f"crossref_{doi}",
            relevance=_relevance_score(title, abstract, qk),
            evidence_weight=_evidence_weight(abstract),
        )
        papers.append(p)
    return papers


async def fetch_crossref(query: str) -> list:
    raw = await _safe_fetch(_fetch_crossref_raw(query), "crossref")
    return relevance_filter(raw, query)


async def fetch_unpaywall_pdf(doi: str) -> Optional[str]:
    if not doi:
        return None
    url = f"https://api.unpaywall.org/v2/{doi}?email={UNPAYWALL_EMAIL}"
    try:
        async with httpx.AsyncClient(timeout=6) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            best = resp.json().get("best_oa_location")
            if best:
                return best.get("url_for_pdf") or best.get("url")
    except Exception:
        pass
    return None


async def enrich_with_unpaywall(papers: list) -> list:
    needs_oa = [p for p in papers if not p.pdf_url and p.doi]
    if not needs_oa:
        return papers
    tasks   = [fetch_unpaywall_pdf(p.doi) for p in needs_oa]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    for paper, result in zip(needs_oa, results):
        if isinstance(result, str):
            paper.pdf_url = result
    return papers


def dedup_and_rank(papers: list, query: str = "", top_n: int = 14) -> list:
    seen: set = set()
    unique = []
    for p in papers:
        key = re.sub(r"\s+", " ", (p.title or "").lower().strip())
        if key and key not in seen:
            seen.add(key)
            unique.append(p)

    if query:
        qk = _keywords(query)
        for p in unique:
            p.relevance = _relevance_score(p.title or "", p.abstract or "", qk)
            p.evidence_weight = _evidence_weight(p.abstract or "")

    def score(p):
        rel      = (p.relevance or 0.0) * 0.45
        cit      = math.log((p.citation_count or 0) + 1) * 0.25
        recency  = max(0, (p.year or 2000) - 2015) * 0.04
        evidence = ((p.evidence_weight or 1.0) - 1.0) * 0.1
        return rel + cit + recency + evidence

    unique.sort(key=score, reverse=True)
    return unique[:top_n]


async def fetch_citation_network(doi: str) -> dict:
    if not doi:
        return {}
    url = f"https://api.crossref.org/works/{doi}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, params={"mailto": UNPAYWALL_EMAIL})
            if resp.status_code != 200:
                return {}
            work = resp.json().get("message", {})
        refs = []
        for r in (work.get("reference") or [])[:12]:
            refs.append({
                "doi":     r.get("DOI", ""),
                "title":   r.get("article-title", "") or r.get("volume-title", ""),
                "year":    r.get("year", ""),
                "journal": r.get("journal-title", ""),
            })
        return {
            "doi":             doi,
            "citation_count":  work.get("is-referenced-by-count", 0),
            "reference_count": work.get("references-count", 0),
            "references":      refs,
        }
    except Exception:
        return {}
