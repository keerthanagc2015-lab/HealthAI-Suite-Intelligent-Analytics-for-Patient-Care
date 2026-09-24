"""
HealthAI - Medical RAG Runtime Adapter
Final production/demo runtime adapter.

Uses the already-built HealthAI RAG artifacts:
- data/processed/rag/chunks/all_rag_chunks.json
- data/processed/rag/embeddings/rag_embeddings.npy
- sentence-transformers/all-MiniLM-L6-v2
- safety threshold = 0.25
- clean extractive grounded answering

No retraining is performed.
"""

from pathlib import Path
from functools import lru_cache
import json
import re

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "rag" / "chunks" / "all_rag_chunks.json"
)
EMBEDDINGS_PATH = (
    PROJECT_ROOT / "data" / "processed" / "rag" / "embeddings" / "rag_embeddings.npy"
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SAFETY_THRESHOLD = 0.25
SENTENCE_THRESHOLD = 0.35
TOP_K = 5
MAX_SENTENCES = 3


def _extract_text(item):
    """Find source text across the project's possible chunk schemas."""
    if not isinstance(item, dict):
        return ""

    for key in ("text", "chunk_text", "content", "page_content", "source_text"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        for key in ("text", "chunk_text", "content", "page_content", "source_text"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def _extract_topic(item):
    if not isinstance(item, dict):
        return "Unknown"
    for key in ("topic", "Topic", "category"):
        value = item.get(key)
        if value:
            return str(value)
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        for key in ("topic", "Topic", "category"):
            value = metadata.get(key)
            if value:
                return str(value)
    return "Unknown"


def _extract_document(item):
    if not isinstance(item, dict):
        return "Unknown"
    for key in ("document_id", "document", "source", "source_id"):
        value = item.get(key)
        if value:
            return str(value)
    metadata = item.get("metadata")
    if isinstance(metadata, dict):
        for key in ("document_id", "document", "source", "source_id"):
            value = metadata.get(key)
            if value:
                return str(value)
    return "Unknown"


def _extract_chunk_id(item, fallback):
    if isinstance(item, dict):
        for key in ("chunk_id", "id"):
            if key in item:
                return item[key]
    return fallback


@lru_cache(maxsize=1)
def _load_rag_assets():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"RAG chunks not found: {CHUNKS_PATH}")
    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"RAG embeddings not found: {EMBEDDINGS_PATH}")

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        for key in ("chunks", "data", "records", "items"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break

    if not isinstance(raw, list):
        raise ValueError("all_rag_chunks.json does not contain a chunk list.")

    records = []
    for i, item in enumerate(raw):
        text = _extract_text(item)
        if not text:
            continue
        records.append(
            {
                "chunk_id": _extract_chunk_id(item, i),
                "document_id": _extract_document(item),
                "topic": _extract_topic(item),
                "text": text,
            }
        )

    embeddings = np.load(EMBEDDINGS_PATH)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    if len(records) != len(embeddings):
        # Defensive alignment: if the aggregate file contains extra metadata
        # records, keep only the aligned prefix rather than silently pairing
        # unrelated vectors.
        n = min(len(records), len(embeddings))
        records = records[:n]
        embeddings = embeddings[:n]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)

    model = SentenceTransformer(EMBEDDING_MODEL)
    return model, records, embeddings


def _split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _select_sentences(question, source_text, model):
    sentences = _split_sentences(source_text)
    if not sentences:
        return []

    q_vec = model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    s_vec = model.encode(
        sentences,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    scores = np.dot(s_vec, q_vec)
    ranked = np.argsort(scores)[::-1]

    selected = []
    seen = set()

    for idx in ranked:
        score = float(scores[idx])
        sentence = sentences[idx]

        if score < SENTENCE_THRESHOLD:
            continue

        normalized = re.sub(r"\W+", " ", sentence.lower()).strip()
        if normalized in seen:
            continue

        # Avoid very tiny fragments.
        if len(sentence.split()) < 6:
            continue

        selected.append(
            {
                "sentence": sentence,
                "score": round(score, 4),
            }
        )
        seen.add(normalized)

        if len(selected) >= MAX_SENTENCES:
            break

    return selected


def _keyword_overlap(question, text):
    q = set(re.findall(r"[a-zA-Z0-9]+", question.lower()))
    t = set(re.findall(r"[a-zA-Z0-9]+", text.lower()))
    if not q:
        return 0.0
    return len(q & t) / len(q)


def answer_medical_question(query: str):
    """
    Runtime question answering.

    Safety behavior:
    - retrieve evidence
    - require top evidence similarity >= 0.25
    - otherwise abstain
    - answer only from retrieved source text
    """
    if not isinstance(query, str) or not query.strip():
        return {
            "status": "fallback",
            "tool": "medical_rag",
            "answer": "Please enter a healthcare-related question.",
        }

    model, records, embeddings = _load_rag_assets()

    q_vec = model.encode(
        [query.strip()],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    semantic_scores = np.dot(embeddings, q_vec)

    # Question-aware reranking: preserve semantic retrieval while giving
    # exact query terms a modest boost.
    candidates = []
    for i, record in enumerate(records):
        semantic = float(semantic_scores[i])
        keyword = _keyword_overlap(query, record["text"])
        combined = 0.70 * semantic + 0.30 * keyword
        candidates.append((combined, semantic, i))

    candidates.sort(reverse=True)

    top = candidates[:TOP_K]

    # Safety gate uses the strongest semantic evidence.
    best_semantic = max(float(x[1]) for x in top) if top else 0.0

    if best_semantic < SAFETY_THRESHOLD:
        return {
            "status": "abstain",
            "tool": "medical_rag",
            "query": query,
            "answer": (
                "I don't have enough information in my current medical "
                "knowledge sources to answer that reliably."
            ),
            "safety_threshold": SAFETY_THRESHOLD,
            "best_similarity": round(best_semantic, 4),
            "grounding_method": "clean_extractive_grounded_rag",
        }

    # Keep the strongest few pieces of evidence.
    selected_records = []
    for combined, semantic, idx in top[:3]:
        if semantic >= SAFETY_THRESHOLD:
            record = dict(records[idx])
            record["similarity"] = round(float(semantic), 4)
            record["rerank_score"] = round(float(combined), 4)
            selected_records.append(record)

    source_text = " ".join(r["text"] for r in selected_records)
    selected_sentences = _select_sentences(query, source_text, model)

    if not selected_sentences:
        return {
            "status": "abstain",
            "tool": "medical_rag",
            "query": query,
            "answer": (
                "I don't have enough information in my current medical "
                "knowledge sources to answer that reliably."
            ),
            "safety_threshold": SAFETY_THRESHOLD,
            "best_similarity": round(best_semantic, 4),
            "grounding_method": "clean_extractive_grounded_rag",
        }

    answer = " ".join(x["sentence"] for x in selected_sentences)

    sources = []
    seen = set()
    for record in selected_records:
        key = (record["document_id"], record["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "document": record["document_id"],
                "topic": record["topic"],
                "chunk_id": record["chunk_id"],
                "similarity": record["similarity"],
            }
        )

    return {
        "status": "success",
        "tool": "medical_rag",
        "query": query,
        "answer": answer,
        "grounding_method": "clean_extractive_grounded_rag",
        "safety_threshold": SAFETY_THRESHOLD,
        "best_similarity": round(best_semantic, 4),
        "sources": sources,
        "safety": {
            "grounded": True,
            "ood_rejection_enabled": True,
            "abstention_supported": True,
        },
    }


def execute_medical_rag(query="", **kwargs):
    return answer_medical_question(query)


if __name__ == "__main__":
    print("=" * 70)
    print("HEALTHAI - MEDICAL RAG RUNTIME ADAPTER")
    print("=" * 70)

    result = execute_medical_rag(
        "What are the symptoms of diabetes?"
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
