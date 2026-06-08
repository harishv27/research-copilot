import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY", "")
UNPAYWALL_EMAIL  = os.getenv("UNPAYWALL_EMAIL", "research@copilot.ai")

EMBED_MODEL      = "text-embedding-3-small"
SYNTHESIS_MODEL  = "gpt-4o"
FAST_MODEL       = "gpt-4o-mini"

MAX_ARXIV_RESULTS        = 12
MAX_SEMANTIC_RESULTS     = 12
MAX_PUBMED_RESULTS       = 10
MAX_CROSSREF_RESULTS     = 10
MAX_PAPERS_FOR_SYNTHESIS = 14
MAX_PAPERS_FOR_CONSENSUS = 16

CHUNK_SIZE           = 600
CHUNK_OVERLAP        = 60
TOP_K_RETRIEVAL      = 12
MAX_REFLECTION_LOOPS = 2
CONFIDENCE_THRESHOLD = 0.55
MAX_SUBQUERIES       = 3
MIN_KEYWORD_OVERLAP  = 0.08
SOURCE_TIMEOUT       = 9.0

CHROMA_PERSIST_DIR   = "./data/chroma_db"
CHROMA_COLLECTION    = "research_papers_v2"
MEMORY_DB_PATH       = "memory/research_memory.db"
MAX_HISTORY_ITEMS    = 100
