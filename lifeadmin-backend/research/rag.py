"""ChromaDB knowledge base (persistent, multilingual, relevance-thresholded).

Two kinds of data live in the same collection:
  * owner == "public"  -> curated official data (ingested by the team via CLI)
  * owner == <user_id> -> a user's own uploaded documents (CV, passport scan...)

Queries always filter by owner, so one user can never retrieve another user's
documents.
"""

import hashlib
import logging
from datetime import date

from . import config

logger = logging.getLogger("lifeadmin.research.rag")


def _embedding_function():
    if config.EMBEDDING_BACKEND == "default":
        return None
    try:
        from chromadb.utils import embedding_functions

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL
        )
    except Exception as exc:
        logger.warning(
            "Multilingual embeddings unavailable (%s); using Chroma default "
            "(English). Install sentence-transformers to enable Urdu support.",
            type(exc).__name__,
        )
        return None


def create_chroma_collection(collection_name=None, persist_path=None):
    """Open (or create) the persistent collection. Returns (client, collection).

    NOTE: if you change the embedding backend/model, delete the chroma_db
    folder and re-ingest - vectors from different models are incompatible.
    """
    import chromadb

    client = chromadb.PersistentClient(path=persist_path or config.CHROMA_PATH)
    kwargs = {
        "name": collection_name or config.COLLECTION_NAME,
        "metadata": {"hnsw:space": "cosine"},
    }
    embedding_function = _embedding_function()
    if embedding_function is not None:
        kwargs["embedding_function"] = embedding_function
    return client, client.get_or_create_collection(**kwargs)


def chunk_text(text, chunk_words=None, overlap_words=None):
    """Split text into overlapping word-based chunks."""
    chunk_words = chunk_words or config.CHUNK_WORDS
    overlap_words = (
        config.CHUNK_OVERLAP_WORDS if overlap_words is None else overlap_words
    )
    if overlap_words >= chunk_words:
        raise ValueError("overlap must be smaller than the chunk size")

    words = (text or "").split()
    if not words:
        return []

    step = chunk_words - overlap_words
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start : start + chunk_words]
        chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return chunks


def _clean_metadata(metadata):
    """Chroma metadata values must be str/int/float/bool (no None)."""
    return {k: v for k, v in metadata.items() if v is not None}


def add_documents(collection, documents, ids, metadatas=None):
    """Insert/update raw chunks (idempotent: same id overwrites)."""
    if metadatas is None:
        metadatas = [{"owner": config.PUBLIC_OWNER} for _ in documents]
    collection.upsert(
        documents=list(documents),
        ids=list(ids),
        metadatas=[_clean_metadata(m) for m in metadatas],
    )


def ingest_text(
    collection,
    text,
    title,
    source_url="",
    source_type="user_document",
    owner=None,
    date_fetched=None,
):
    """Chunk and store one document. Re-ingesting the same source replaces it.

    Returns the number of chunks stored.
    """
    chunks = chunk_text(text)
    if not chunks:
        return 0

    owner = owner or config.PUBLIC_OWNER
    source_key = hashlib.sha1(
        f"{owner}|{source_url or title}".encode("utf-8")
    ).hexdigest()[:16]

    try:  # remove the previous version of this document, if any
        collection.delete(where={"source_key": source_key})
    except Exception as exc:
        logger.debug("No previous version removed: %s", type(exc).__name__)

    metadata = {
        "owner": owner,
        "title": title,
        "source_url": source_url,
        "source_type": source_type,
        "source_key": source_key,
        "date_fetched": date_fetched or date.today().isoformat(),
    }
    add_documents(
        collection,
        chunks,
        [f"{source_key}-{i}" for i in range(len(chunks))],
        [dict(metadata, chunk_index=i) for i in range(len(chunks))],
    )
    return len(chunks)


def search_documents(
    collection, query, n_results=None, max_distance=None, owner=None
):
    """Semantic search restricted to public data + the given user's data.

    Returns a list of {"text", "metadata", "distance"}; hits farther than
    `max_distance` are dropped, so an irrelevant knowledge base returns [].
    """
    n_results = n_results or config.RAG_N_RESULTS
    max_distance = config.RAG_MAX_DISTANCE if max_distance is None else max_distance

    try:
        total = collection.count()
        if total == 0:
            return []
        n_results = min(n_results, total)
    except Exception:  # count() is only an optimisation
        pass

    owners = [config.PUBLIC_OWNER] + ([owner] if owner else [])
    where = {"owner": {"$in": owners}}

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    hits = []
    for i, text in enumerate(documents):
        distance = distances[i] if i < len(distances) else None
        if distance is None or distance > max_distance:
            continue
        hits.append(
            {
                "text": text,
                "metadata": (metadatas[i] if i < len(metadatas) else None) or {},
                "distance": distance,
            }
        )
    return hits


def pdf_contains_answer(collection, query, owner=None):
    """Backwards-compatible helper: (found, context_text)."""
    hits = search_documents(collection, query, owner=owner)
    if not hits:
        return False, ""
    return True, "\n\n".join(hit["text"] for hit in hits)
