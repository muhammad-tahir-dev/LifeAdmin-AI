"""Central configuration for the LifeAdmin research module.

Everything tunable lives here so teammates never have to edit pipeline code.
Values can be overridden with environment variables (see .env.example).
"""

import os
from pathlib import Path

try:  # python-dotenv is optional; plain environment variables also work
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent.parent

# --- LLM (Groq) ---------------------------------------------------------
# Groq's catalogue changes; list what your key can use with client.models.list().
# gpt-oss models are reasoning models: reasoning tokens count towards the token
# limit, so keep effort low and the limit generous or the visible answer is empty.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")  # low|medium|high
GROQ_MAX_COMPLETION_TOKENS = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "3000"))

# --- Vector store / RAG -------------------------------------------------
CHROMA_PATH = os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db"))
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "lifeadmin_documents")
# "multilingual" -> sentence-transformers model (handles Urdu / Roman Urdu better)
# "default"      -> Chroma's built-in English MiniLM (lighter, no torch needed)
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "multilingual")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
)
RAG_N_RESULTS = int(os.getenv("RAG_N_RESULTS", "3"))
# Cosine distance: 0 = identical, larger = less similar. Tune on real queries.
RAG_MAX_DISTANCE = float(os.getenv("RAG_MAX_DISTANCE", "0.55"))
CHUNK_WORDS = 220
CHUNK_OVERLAP_WORDS = 40
PUBLIC_OWNER = "public"  # owner tag for curated/official knowledge-base data

# A curated official chunk older than this is considered stale -> web re-check.
KB_MAX_AGE_DAYS = int(os.getenv("KB_MAX_AGE_DAYS", "30"))

# --- Web research -------------------------------------------------------
MAX_WEB_SOURCES = 5
MAX_SOURCE_CHARS = 1500  # per-source text sent to the LLM (token/rate-limit control)

# Domains are matched against the *hostname* (exact match or subdomain), never
# as a substring of the URL.
OFFICIAL_DOMAINS = (
    "gov.pk",
    "gop.pk",
    "gov.uk",
    "gov",  # US federal (.gov is restricted to US government bodies)
    "usa.gov",
    "canada.ca",
    "gc.ca",
    "gov.au",
    "govt.nz",
    "gov.in",
    "gov.ae",
    "gov.sg",
    "gov.za",
    "gob.mx",
    "bund.de",
    "admin.ch",
    "make-it-in-germany.com",
    "germany.info",
)

COMMUNITY_DOMAINS = (
    "youtube.com",
    "youtu.be",
    "facebook.com",
    "reddit.com",
    "quora.com",
    "x.com",
    "twitter.com",
    "tiktok.com",
)

# Used to restrict the *first* Tavily pass to known official Pakistani sites.
# If that pass finds nothing official, the pipeline falls back to an open search
# and relies on source scoring. Verify/extend this list for your flagship cases.
DEFAULT_OFFICIAL_SEARCH_DOMAINS = [
    "nadra.gov.pk",
    "dgip.gov.pk",
    "mofa.gov.pk",
    "interior.gov.pk",
    "hec.gov.pk",
    "fbr.gov.pk",
    "pakistan.gov.pk",
    "punjab.gov.pk",
]

# --- Source trust -------------------------------------------------------
SOURCE_CONFIDENCE = {
    "official": 1.0,
    "user_document": 0.9,
    "other": 0.6,
    "community": 0.3,
}
