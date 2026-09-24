"""
HealthAI - Improved RAG Answer Generation

Step 38H

Purpose:
Improve the baseline RAG generation from Step 38G.

Pipeline:

Question
   ↓
Semantic Retrieval
   ↓
Top-5 Evidence
   ↓
Evidence Filtering
   ↓
Grounded Prompt
   ↓
FLAN-T5 Base
   ↓
Answer
   ↓
Source Attribution
   ↓
Abstention

Baseline:
    google/flan-t5-small

Improved:
    google/flan-t5-base
"""

from pathlib import Path
import json
import csv
import re

import numpy as np

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "embeddings"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "rag"
    / "generation"
    / "improved"
)

EMBEDDING_FILE = (
    EMBEDDING_DIR
    / "rag_embeddings.npy"
)

METADATA_FILE = (
    EMBEDDING_DIR
    / "rag_embedding_metadata.json"
)

RESULTS_FILE = (
    OUTPUT_DIR
    / "rag_improved_generation_results.json"
)

EVALUATION_FILE = (
    OUTPUT_DIR
    / "rag_improved_generation_evaluation.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "rag_improved_generation_validation.json"
)


# ============================================================
# MODELS
# ============================================================

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

GENERATION_MODEL_NAME = (
    "google/flan-t5-base"
)


# ============================================================
# RAG SETTINGS
# ============================================================

TOP_K = 5

MIN_RETRIEVAL_SCORE = 0.45

MAX_INPUT_TOKENS = 768

MAX_NEW_TOKENS = 220


# ============================================================
# TEST QUESTIONS
# ============================================================

TEST_QUESTIONS = [

    {
        "question":
            "What are the symptoms of diabetes?",
        "expected_topics":
            ["Diabetes", "Type 2 Diabetes Self-Care"],
        "answerable":
            True,
    },

    {
        "question":
            "How can diabetes be managed?",
        "expected_topics":
            ["Diabetes", "Type 2 Diabetes Self-Care"],
        "answerable":
            True,
    },

    {
        "question":
            "What is high blood pressure?",
        "expected_topics":
            ["Hypertension"],
        "answerable":
            True,
    },

    {
        "question":
            "How can high blood pressure be prevented?",
        "expected_topics":
            ["Hypertension"],
        "answerable":
            True,
    },

    {
        "question":
            "What are noncommunicable diseases?",
        "expected_topics":
            ["Noncommunicable Diseases"],
        "answerable":
            True,
    },

    {
        "question":
            "How can someone cope with a chronic illness?",
        "expected_topics":
            ["Living With Chronic Illness"],
        "answerable":
            True,
    },

    {
        "question":
            "What is the recommended treatment for appendicitis?",
        "expected_topics":
            [],
        "answerable":
            False,
    },
]


# ============================================================
# SAVE JSON
# ============================================================

def save_json(data, path):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    query_vector,
    document_vectors
):

    query_norm = np.linalg.norm(
        query_vector
    )

    document_norms = np.linalg.norm(
        document_vectors,
        axis=1
    )

    if query_norm == 0:

        raise ValueError(
            "Query embedding is zero."
        )

    if np.any(
        document_norms == 0
    ):

        raise ValueError(
            "Vector store contains zero vectors."
        )

    query_vector = (
        query_vector
        / query_norm
    )

    document_vectors = (
        document_vectors
        / document_norms[:, None]
    )

    return np.dot(
        document_vectors,
        query_vector
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_chunks(
    question,
    embedding_model,
    embeddings,
    metadata,
    top_k
):

    query_embedding = (
        embedding_model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )

    indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for rank, index in enumerate(
        indices,
        start=1
    ):

        item = metadata[
            int(index)
        ]

        results.append(
            {
                "rank":
                    rank,

                "embedding_index":
                    int(index),

                "similarity_score":
                    float(
                        scores[index]
                    ),

                "chunk_id":
                    item["chunk_id"],

                "document_id":
                    item["document_id"],

                "topic":
                    item["topic"],

                "organization":
                    item["organization"],

                "authority":
                    item["authority"],

                "source_type":
                    item["source_type"],

                "source_url":
                    item["source_url"],

                "text":
                    item["text"],
            }
        )

    return results


# ============================================================
# EVIDENCE FILTERING
# ============================================================

def filter_evidence(
    retrieved_chunks
):

    if not retrieved_chunks:

        return []

    top_score = (
        retrieved_chunks[0]
        ["similarity_score"]
    )

    # If the best result itself is weak,
    # don't attempt answer generation.
    if top_score < MIN_RETRIEVAL_SCORE:

        return []

    filtered = []

    for chunk in retrieved_chunks:

        score = chunk[
            "similarity_score"
        ]

        # Keep reasonably relevant evidence.
        #
        # We also require the chunk to be
        # relatively close to the strongest result.
        if (
            score >= MIN_RETRIEVAL_SCORE
            and score >= top_score * 0.75
        ):

            filtered.append(
                chunk
            )

    return filtered


# ============================================================
# CONTEXT BUILDING
# ============================================================

def build_context(
    evidence
):

    context = []

    for index, chunk in enumerate(
        evidence,
        start=1
    ):

        context.append(
            f"""
[EVIDENCE {index}]

Organization:
{chunk['organization']}

Topic:
{chunk['topic']}

Authority:
{chunk['authority']}

Source:
{chunk['source_url']}

Content:
{chunk['text']}
""".strip()
        )

    return "\n\n".join(
        context
    )


# ============================================================
# PROMPT
# ============================================================

def build_prompt(
    question,
    context
):

    return f"""
You are HealthAI, a careful healthcare
information assistant.

Your task is to answer the user's question
using ONLY the evidence provided below.

IMPORTANT RULES:

1. Use only information explicitly supported
   by the evidence.

2. Do not use outside medical knowledge.

3. Do not diagnose the user.

4. Do not invent symptoms, treatments,
   causes, statistics, or recommendations.

5. If the evidence does not answer the
   question, say exactly:

"I don't have enough information in my
current medical knowledge sources to answer
that question reliably."

6. Answer the actual question directly.

7. Prefer a short paragraph or a few
   bullet points.

8. Do not mention internal retrieval,
   embeddings, vector databases, or models.

9. Do not copy irrelevant information from
   the evidence.

10. At the end, write:

Sources:
- [EVIDENCE X]

where X corresponds to the evidence used.

USER QUESTION:
{question}

EVIDENCE:
{context}

ANSWER:
""".strip()


# ============================================================
# ANSWER CLEANING
# ============================================================

def clean_answer(
    answer
):

    answer = answer.strip()

    answer = re.sub(
        r"\n{3,}",
        "\n\n",
        answer
    )

    return answer


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    question,
    evidence,
    tokenizer,
    model
):

    if not evidence:

        return (
            "I don't have enough information "
            "in my current medical knowledge "
            "sources to answer that question "
            "reliably."
        )

    context = build_context(
        evidence
    )

    prompt = build_prompt(
        question,
        context
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return clean_answer(
        answer
    )


# ============================================================
# START
# ============================================================

print("=" * 75)
print("HEALTHAI - IMPROVED RAG ANSWER GENERATION")
print("=" * 75)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

print("\nLoading embeddings...")

embeddings = np.load(
    EMBEDDING_FILE
)

print(
    f"✓ Embeddings loaded: "
    f"{embeddings.shape}"
)


# ============================================================
# LOAD METADATA
# ============================================================

print(
    "\nLoading metadata..."
)

with open(
    METADATA_FILE,
    "r",
    encoding="utf-8"
) as f:

    metadata_payload = json.load(f)


metadata = metadata_payload[
    "metadata"
]

print(
    f"✓ Metadata loaded: "
    f"{len(metadata)}"
)


# ============================================================
# VALIDATE
# ============================================================

if len(embeddings) != len(metadata):

    raise ValueError(
        "Embedding count and metadata "
        "count do not match."
    )


if not np.isfinite(
    embeddings
).all():

    raise ValueError(
        "Embedding matrix contains "
        "NaN or infinite values."
    )


print(
    "✓ Vector store validation passed."
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print(
    "\nLoading embedding model..."
)

embedding_model = (
    SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )
)

print(
    "✓ Embedding model loaded."
)


# ============================================================
# LOAD GENERATION MODEL
# ============================================================

print(
    "\nLoading improved generation model..."
)

print(
    f"Model: {GENERATION_MODEL_NAME}"
)


tokenizer = (
    AutoTokenizer.from_pretrained(
        GENERATION_MODEL_NAME
    )
)


generation_model = (
    AutoModelForSeq2SeqLM.from_pretrained(
        GENERATION_MODEL_NAME
    )
)


generation_model.eval()


print(
    "✓ FLAN-T5 Base tokenizer loaded."
)

print(
    "✓ FLAN-T5 Base model loaded."
)


# ============================================================
# RUN TESTS
# ============================================================

results = []

evaluation_rows = []


print(
    "\n" + "=" * 75
)

print(
    "IMPROVED RAG TESTS"
)

print(
    "=" * 75
)


for question_number, test_case in enumerate(
    TEST_QUESTIONS,
    start=1
):

    question = test_case[
        "question"
    ]

    expected_topics = test_case[
        "expected_topics"
    ]

    expected_answerable = test_case[
        "answerable"
    ]


    print(
        "\n" + "-" * 75
    )

    print(
        f"Question {question_number}: "
        f"{question}"
    )


    # --------------------------------------------------------
    # RETRIEVE TOP 5
    # --------------------------------------------------------

    retrieved = retrieve_chunks(
        question,
        embedding_model,
        embeddings,
        metadata,
        TOP_K
    )


    # --------------------------------------------------------
    # FILTER EVIDENCE
    # --------------------------------------------------------

    evidence = filter_evidence(
        retrieved
    )


    retrieved_topics = [
        item["topic"]
        for item in retrieved
    ]


    evidence_topics = [
        item["topic"]
        for item in evidence
    ]


    top_score = (
        retrieved[0]
        ["similarity_score"]
        if retrieved
        else 0.0
    )


    expected_topic_found = any(
        topic in expected_topics
        for topic in retrieved_topics
    )


    # --------------------------------------------------------
    # GENERATE
    # --------------------------------------------------------

    answer = generate_answer(
        question,
        evidence,
        tokenizer,
        generation_model
    )


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"\nTop similarity: "
        f"{top_score:.4f}"
    )

    print(
        f"Retrieved topics: "
        f"{retrieved_topics}"
    )

    print(
        f"Evidence used: "
        f"{len(evidence)}"
    )

    print(
        "\nGenerated answer:"
    )

    print(answer)


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    results.append(
        {
            "question_number":
                question_number,

            "question":
                question,

            "expected_topics":
                expected_topics,

            "expected_answerable":
                expected_answerable,

            "top_similarity":
                top_score,

            "retrieved_topics":
                retrieved_topics,

            "evidence_topics":
                evidence_topics,

            "expected_topic_found":
                expected_topic_found,

            "answer":
                answer,

            "evidence":
                evidence,

            "all_retrieved_chunks":
                retrieved,
        }
    )


    evaluation_rows.append(
        {
            "question_number":
                question_number,

            "question":
                question,

            "expected_answerable":
                expected_answerable,

            "expected_topics":
                "|".join(
                    expected_topics
                ),

            "top_similarity":
                top_score,

            "retrieved_topic":
                (
                    retrieved[0]["topic"]
                    if retrieved
                    else ""
                ),

            "expected_topic_found":
                expected_topic_found,

            "evidence_chunks":
                len(evidence),

            "answer_length":
                len(answer),

            "abstained":
                (
                    "I don't have enough information"
                    in answer
                ),
        }
    )


# ============================================================
# METRICS
# ============================================================

answerable = [
    r
    for r in results
    if r["expected_answerable"]
]


unanswerable = [
    r
    for r in results
    if not r["expected_answerable"]
]


topic_hits = sum(
    r["expected_topic_found"]
    for r in answerable
)


if answerable:

    topic_hit_rate = (
        topic_hits
        / len(answerable)
    )

else:

    topic_hit_rate = 0.0


abstentions = sum(
    "I don't have enough information"
    in r["answer"]
    for r in unanswerable
)


if unanswerable:

    abstention_rate = (
        abstentions
        / len(unanswerable)
    )

else:

    abstention_rate = 0.0


# ============================================================
# SAVE RESULTS
# ============================================================

save_json(
    {
        "embedding_model":
            EMBEDDING_MODEL_NAME,

        "generation_model":
            GENERATION_MODEL_NAME,

        "top_k":
            TOP_K,

        "minimum_retrieval_score":
            MIN_RETRIEVAL_SCORE,

        "answerable_topic_hit_rate":
            topic_hit_rate,

        "out_of_corpus_abstention_rate":
            abstention_rate,

        "results":
            results,
    },
    RESULTS_FILE
)


# ============================================================
# SAVE CSV
# ============================================================

with open(
    EVALUATION_FILE,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "question_number",
            "question",
            "expected_answerable",
            "expected_topics",
            "top_similarity",
            "retrieved_topic",
            "expected_topic_found",
            "evidence_chunks",
            "answer_length",
            "abstained",
        ]
    )

    writer.writeheader()

    writer.writerows(
        evaluation_rows
    )


# ============================================================
# VALIDATION
# ============================================================

validation = {

    "embedding_count":
        int(len(embeddings)),

    "embedding_dimension":
        int(embeddings.shape[1]),

    "metadata_count":
        int(len(metadata)),

    "generation_model_loaded":
        True,

    "test_question_count":
        len(results),

    "answerable_question_count":
        len(answerable),

    "answerable_topic_hit_rate":
        topic_hit_rate,

    "out_of_corpus_question_count":
        len(unanswerable),

    "out_of_corpus_abstention_rate":
        abstention_rate,

    "vector_store_valid":
        True,

    "generation_completed":
        len(results) == len(
            TEST_QUESTIONS
        ),

    "ready_for_generation_comparison":
        True,
}


save_json(
    validation,
    VALIDATION_FILE
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "STEP 38H - IMPROVED RAG GENERATION COMPLETED"
)

print(
    "=" * 75
)

print(
    f"\nGeneration model:"
    f" {GENERATION_MODEL_NAME}"
)

print(
    f"Top-K retrieval:"
    f" {TOP_K}"
)

print(
    f"Answerable topic hit rate:"
    f" {topic_hit_rate:.2%}"
)

print(
    f"Out-of-corpus abstention rate:"
    f" {abstention_rate:.2%}"
)

print(
    "\nSaved:"
)

print(
    RESULTS_FILE
)

print(
    EVALUATION_FILE
)

print(
    VALIDATION_FILE
)

print(
    "\n✓ Improved RAG generation completed."
)