"""
HEALTHAI - STEP 38L
Safety-Aware RAG Context Builder

Purpose:
- Apply the validated retrieval relevance threshold.
- Filter weak evidence before generation.
- Build clean grounded context for the RAG generator.
- Explicitly abstain when evidence is insufficient.
"""

import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "retrieval"
    / "improved"
    / "rag_improved_retrieval_results.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "context"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


OUTPUT_JSON = OUTPUT_DIR / "rag_safety_aware_contexts.json"
OUTPUT_CSV = OUTPUT_DIR / "rag_safety_aware_context_evaluation.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "rag_safety_aware_context_summary.json"


# ============================================================
# LOCKED SAFETY CONFIGURATION
# ============================================================

RELEVANCE_THRESHOLD = 0.25

MAX_CONTEXT_CHUNKS = 3


# ============================================================
# LOAD RETRIEVAL RESULTS
# ============================================================

print("=" * 75)
print("HEALTHAI - SAFETY-AWARE RAG CONTEXT BUILDER")
print("=" * 75)

print("\nLoading question-aware retrieval results...")

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

if isinstance(data, dict):

    if "results" in data:
        results = data["results"]

    elif "test_results" in data:
        results = data["test_results"]

    else:
        raise ValueError(
            "Could not find retrieval results. "
            "Expected 'results' or 'test_results'."
        )

else:
    results = data

print("✓ Retrieval results loaded.")
print(f"Test questions: {len(results)}")


# ============================================================
# HELPERS
# ============================================================

def get_final_evidence(item):
    evidence = item.get("final_evidence", [])

    if not isinstance(evidence, list):
        return []

    return evidence


def get_score(evidence):
    """
    Retrieve the final reranking score.
    Supports multiple possible field names.
    """

    for key in [
        "rerank_score",
        "final_score",
        "score",
        "similarity",
        "semantic_score",
    ]:

        value = evidence.get(key)

        if value is not None:

            try:
                return float(value)

            except (TypeError, ValueError):
                pass

    return None


def get_topic(evidence):
    """
    Retrieve topic from direct or nested metadata.
    """

    topic = evidence.get("topic")

    if topic is None:

        metadata = evidence.get("metadata", {})

        if isinstance(metadata, dict):
            topic = metadata.get("topic")

    if topic is None:
        return ""

    return str(topic)


def get_text(evidence):
    """
    Retrieve chunk text.
    """

    text = evidence.get("text")

    if text is None:
        text = evidence.get("chunk_text")

    if text is None:
        text = ""

    return str(text).strip()


def get_source(evidence):
    """
    Retrieve source information.
    """

    metadata = evidence.get("metadata", {})

    if isinstance(metadata, dict):

        source_id = metadata.get("source_id")

        if source_id:
            return str(source_id)

        source_url = metadata.get("source_url")

        if source_url:
            return str(source_url)

    source_id = evidence.get("source_id")

    if source_id:
        return str(source_id)

    return "unknown"


# ============================================================
# BUILD SAFETY-AWARE CONTEXTS
# ============================================================

print("\n" + "=" * 75)
print("BUILDING SAFETY-AWARE CONTEXT")
print("=" * 75)

context_results = []
evaluation_rows = []

accepted_count = 0
abstained_count = 0

total_evidence_before = 0
total_evidence_after = 0


for item in results:

    question_number = item.get("question_number")
    question = item.get("question")

    answerable = bool(
        item.get("answerable", True)
    )

    expected_topics = item.get(
        "expected_topics",
        []
    )

    if isinstance(expected_topics, str):
        expected_topics = [expected_topics]

    evidence = get_final_evidence(item)

    total_evidence_before += len(evidence)

    # --------------------------------------------------------
    # SCORE AND FILTER
    # --------------------------------------------------------

    scored_evidence = []

    for ev in evidence:

        score = get_score(ev)

        if score is None:
            continue

        text = get_text(ev)

        if not text:
            continue

        topic = get_topic(ev)

        source = get_source(ev)

        scored_evidence.append(
            {
                "score": score,
                "topic": topic,
                "source": source,
                "text": text,
                "chunk_id": ev.get(
                    "chunk_id",
                    ev.get("id", "")
                ),
            }
        )

    accepted_evidence = [
        ev
        for ev in scored_evidence
        if ev["score"] >= RELEVANCE_THRESHOLD
    ]

    # Highest score first.
    accepted_evidence = sorted(
        accepted_evidence,
        key=lambda x: x["score"],
        reverse=True
    )

    # Limit context size.
    accepted_evidence = accepted_evidence[
        :MAX_CONTEXT_CHUNKS
    ]

    total_evidence_after += len(
        accepted_evidence
    )

    # --------------------------------------------------------
    # ABSTENTION DECISION
    # --------------------------------------------------------

    if len(accepted_evidence) == 0:

        decision = "ABSTAIN"

        answerable_by_retrieval = False

        context = ""

        abstained_count += 1

    else:

        decision = "ACCEPT"

        answerable_by_retrieval = True

        accepted_count += 1

        # ----------------------------------------------------
        # BUILD GROUNDED CONTEXT
        # ----------------------------------------------------

        context_parts = []

        for rank, ev in enumerate(
            accepted_evidence,
            start=1
        ):

            context_parts.append(
                (
                    f"[Evidence {rank}]\n"
                    f"Source: {ev['source']}\n"
                    f"Topic: {ev['topic']}\n"
                    f"Relevance score: "
                    f"{ev['score']:.4f}\n"
                    f"Content:\n"
                    f"{ev['text']}"
                )
            )

        context = "\n\n".join(
            context_parts
        )

    # --------------------------------------------------------
    # STORE RESULT
    # --------------------------------------------------------

    context_results.append(
        {
            "question_number": question_number,
            "question": question,
            "expected_topics": expected_topics,
            "original_answerable": answerable,
            "retrieval_decision": decision,
            "answerable_by_retrieval": (
                answerable_by_retrieval
            ),
            "relevance_threshold": (
                RELEVANCE_THRESHOLD
            ),
            "accepted_evidence_count": len(
                accepted_evidence
            ),
            "context": context,
            "evidence": accepted_evidence,
        }
    )

    evaluation_rows.append(
        {
            "question_number": question_number,
            "question": question,
            "original_answerable": answerable,
            "retrieval_decision": decision,
            "accepted_evidence_count": len(
                accepted_evidence
            ),
            "top_score": (
                accepted_evidence[0]["score"]
                if accepted_evidence
                else None
            ),
            "top_topic": (
                accepted_evidence[0]["topic"]
                if accepted_evidence
                else ""
            ),
            "threshold": RELEVANCE_THRESHOLD,
        }
    )


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        context_results,
        f,
        indent=2,
        ensure_ascii=False
    )


evaluation_df = pd.DataFrame(
    evaluation_rows
)

evaluation_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

summary = {

    "step": "38L",

    "purpose": (
        "Build safety-aware grounded context "
        "for RAG generation."
    ),

    "test_questions": len(results),

    "relevance_threshold": (
        RELEVANCE_THRESHOLD
    ),

    "max_context_chunks": (
        MAX_CONTEXT_CHUNKS
    ),

    "accepted_questions": accepted_count,

    "abstained_questions": abstained_count,

    "acceptance_rate": (
        accepted_count / len(results)
        if results
        else 0.0
    ),

    "abstention_rate": (
        abstained_count / len(results)
        if results
        else 0.0
    ),

    "total_evidence_before_filtering": (
        total_evidence_before
    ),

    "total_evidence_after_filtering": (
        total_evidence_after
    ),

    "files": {
        "contexts": str(OUTPUT_JSON),
        "evaluation": str(OUTPUT_CSV),
        "summary": str(OUTPUT_SUMMARY),
    },
}


with open(
    OUTPUT_SUMMARY,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 75)
print("SAFETY-AWARE CONTEXT RESULTS")
print("=" * 75)

for row in evaluation_rows:

    print(
        f"\nQ{int(row['question_number'])}: "
        f"{row['question']}"
    )

    print(
        f"Decision: "
        f"{row['retrieval_decision']}"
    )

    print(
        f"Top score: "
        f"{row['top_score']}"
    )

    print(
        f"Top topic: "
        f"{row['top_topic'] or 'NONE'}"
    )

    print(
        f"Accepted evidence: "
        f"{row['accepted_evidence_count']}"
    )


print("\n" + "=" * 75)
print("SUMMARY")
print("=" * 75)

print(
    f"\nRelevance threshold: "
    f"{RELEVANCE_THRESHOLD}"
)

print(
    f"Accepted questions: "
    f"{accepted_count}/{len(results)}"
)

print(
    f"Abstained questions: "
    f"{abstained_count}/{len(results)}"
)

print(
    f"Evidence before filtering: "
    f"{total_evidence_before}"
)

print(
    f"Evidence after filtering: "
    f"{total_evidence_after}"
)


print("\n" + "=" * 75)
print("STEP 38L - CONTEXT BUILDER COMPLETED")
print("=" * 75)

print("\nSaved:")
print(OUTPUT_JSON)
print(OUTPUT_CSV)
print(OUTPUT_SUMMARY)