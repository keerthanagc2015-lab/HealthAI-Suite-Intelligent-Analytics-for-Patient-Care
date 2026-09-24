"""
HealthAI - RAG Answer Generation

Step 38G:
Generate grounded healthcare answers using retrieved RAG context.

Pipeline:

User question
      ↓
Query embedding
      ↓
Semantic retrieval
      ↓
Top-K evidence chunks
      ↓
Grounded prompt
      ↓
Local language model
      ↓
Answer + sources

Input:
    data/processed/rag/embeddings/rag_embeddings.npy
    data/processed/rag/embeddings/rag_embedding_metadata.json

Output:
    data/processed/rag/generation/
        rag_generation_results.json
        rag_generation_evaluation.csv
        rag_generation_validation.json
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
# CONFIGURATION
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
    / "rag_generation_results.json"
)

EVALUATION_FILE = (
    OUTPUT_DIR
    / "rag_generation_evaluation.csv"
)

VALIDATION_FILE = (
    OUTPUT_DIR
    / "rag_generation_validation.json"
)


# Sentence embedding model used in Step 38E/38F.
EMBEDDING_MODEL_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Small instruction-following model suitable
# for local CPU experimentation.
GENERATION_MODEL_NAME = (
    "google/flan-t5-small"
)


TOP_K = 3

# Minimum retrieval similarity required before
# using retrieved evidence for generation.
MIN_RETRIEVAL_SCORE = 0.45

MAX_INPUT_TOKENS = 512

MAX_NEW_TOKENS = 180


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

    # Deliberately outside the current corpus.
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
# HELPERS
# ============================================================

def save_json(data, path):
    """Save JSON with readable formatting."""

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


def cosine_similarity(
    query_vector,
    document_vectors
):
    """
    Calculate cosine similarity.

    Embeddings are normalized, so this is
    effectively a dot product.
    """

    query_norm = np.linalg.norm(
        query_vector
    )

    if query_norm == 0:
        raise ValueError(
            "Query embedding is zero."
        )

    document_norms = np.linalg.norm(
        document_vectors,
        axis=1
    )

    if np.any(document_norms == 0):
        raise ValueError(
            "Vector store contains zero vectors."
        )

    query_vector = (
        query_vector / query_norm
    )

    document_vectors = (
        document_vectors
        / document_norms[:, None]
    )

    return np.dot(
        document_vectors,
        query_vector
    )


def retrieve_context(
    question,
    embedding_model,
    embeddings,
    metadata,
    top_k=3
):
    """
    Retrieve the top-k evidence chunks.
    """

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

    top_indices = np.argsort(
        scores
    )[::-1][:top_k]

    results = []

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        item = metadata[int(index)]

        results.append(
            {
                "rank":
                    rank,

                "embedding_index":
                    int(index),

                "similarity_score":
                    float(scores[index]),

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


def build_context(
    retrieved_chunks
):
    """
    Build the evidence context supplied to the LLM.
    """

    context_parts = []

    for result in retrieved_chunks:

        context_parts.append(
            (
                f"[Source {result['rank']}]\n"
                f"Organization: "
                f"{result['organization']}\n"
                f"Topic: "
                f"{result['topic']}\n"
                f"URL: "
                f"{result['source_url']}\n"
                f"Content:\n"
                f"{result['text']}"
            )
        )

    return "\n\n".join(
        context_parts
    )


def build_prompt(
    question,
    context
):
    """
    Construct a healthcare-safe grounded prompt.
    """

    return f"""
You are HealthAI, a healthcare information assistant.

Answer the user's question using ONLY the
provided medical evidence.

Do not introduce medical facts that are not
supported by the evidence.

If the evidence does not contain enough
information to answer the question, say:

"I don't have enough information in my
current medical knowledge sources to answer
that question reliably."

Do not diagnose the user.

Do not claim to replace a healthcare professional.

Give a concise, clear informational answer.

After the answer, include:

Sources:
- source name and URL

User question:
{question}

Medical evidence:
{context}

Answer:
""".strip()


def clean_generated_answer(
    answer
):
    """
    Remove accidental prompt fragments from
    generated output.
    """

    answer = answer.strip()

    # Remove excessive whitespace.
    answer = re.sub(
        r"\n{3,}",
        "\n\n",
        answer
    )

    return answer


def generate_answer(
    question,
    retrieved_chunks,
    tokenizer,
    model
):
    """
    Generate a grounded answer from retrieved evidence.
    """

    # --------------------------------------------------------
    # Retrieval confidence gate
    # --------------------------------------------------------

    if not retrieved_chunks:

        return (
            "I don't have enough information in my "
            "current medical knowledge sources to "
            "answer that question reliably."
        )

    top_score = retrieved_chunks[0][
        "similarity_score"
    ]

    if top_score < MIN_RETRIEVAL_SCORE:

        return (
            "I don't have enough information in my "
            "current medical knowledge sources to "
            "answer that question reliably."
        )

    # --------------------------------------------------------
    # Build evidence context
    # --------------------------------------------------------

    context = build_context(
        retrieved_chunks
    )

    prompt = build_prompt(
        question,
        context
    )

    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS
    )

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=2
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return clean_generated_answer(
        answer
    )


# ============================================================
# HEADER
# ============================================================

print("=" * 75)
print("HEALTHAI - RAG ANSWER GENERATION")
print("=" * 75)


# ============================================================
# CHECK INPUTS
# ============================================================

if not EMBEDDING_FILE.exists():

    raise FileNotFoundError(
        f"Embedding file not found:\n"
        f"{EMBEDDING_FILE}"
    )


if not METADATA_FILE.exists():

    raise FileNotFoundError(
        f"Metadata file not found:\n"
        f"{METADATA_FILE}"
    )


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD VECTOR STORE
# ============================================================

print("\nLoading embeddings...")

embeddings = np.load(
    EMBEDDING_FILE
)

print(
    f"✓ Embeddings loaded: "
    f"{embeddings.shape}"
)


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
# VALIDATE VECTOR STORE
# ============================================================

if len(embeddings) != len(metadata):

    raise ValueError(
        "Embedding and metadata counts differ."
    )


if embeddings.ndim != 2:

    raise ValueError(
        "Embeddings must be 2-dimensional."
    )


if not np.isfinite(
    embeddings
).all():

    raise ValueError(
        "Embeddings contain invalid values."
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
    "\nLoading generation model..."
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
    "✓ Generation tokenizer loaded."
)

print(
    "✓ Generation model loaded."
)


# ============================================================
# RUN RAG QUESTIONS
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "RAG ANSWER GENERATION TESTS"
)

print(
    "=" * 75
)


generation_results = []

evaluation_rows = []


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
    # Retrieval
    # --------------------------------------------------------

    retrieved_chunks = retrieve_context(
        question=question,
        embedding_model=embedding_model,
        embeddings=embeddings,
        metadata=metadata,
        top_k=TOP_K
    )


    retrieved_topics = [
        item["topic"]
        for item in retrieved_chunks
    ]


    top_score = (
        retrieved_chunks[0][
            "similarity_score"
        ]
        if retrieved_chunks
        else 0.0
    )


    expected_topic_found = any(
        topic in expected_topics
        for topic in retrieved_topics
    )


    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        tokenizer=tokenizer,
        model=generation_model
    )


    # --------------------------------------------------------
    # Print retrieval information
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
        "\nGenerated answer:"
    )

    print(answer)


    # --------------------------------------------------------
    # Store results
    # --------------------------------------------------------

    generation_results.append(
        {
            "question_number":
                question_number,

            "question":
                question,

            "expected_topics":
                expected_topics,

            "expected_answerable":
                expected_answerable,

            "retrieved_topics":
                retrieved_topics,

            "expected_topic_found":
                expected_topic_found,

            "top_similarity":
                top_score,

            "answer":
                answer,

            "retrieved_chunks":
                retrieved_chunks,
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

            "top_1_topic":
                (
                    retrieved_chunks[0][
                        "topic"
                    ]
                    if retrieved_chunks
                    else ""
                ),

            "expected_topic_found":
                expected_topic_found,

            "answer_length":
                len(answer),
        }
    )


# ============================================================
# EVALUATION
# ============================================================

answerable_results = [
    result
    for result in generation_results
    if result["expected_answerable"]
]


answerable_topic_hits = sum(
    result["expected_topic_found"]
    for result in answerable_results
)


answerable_count = len(
    answerable_results
)


if answerable_count > 0:

    answerable_topic_hit_rate = (
        answerable_topic_hits
        / answerable_count
    )

else:

    answerable_topic_hit_rate = 0.0


# ============================================================
# OUT-OF-CORPUS TEST
# ============================================================

out_of_corpus_results = [
    result
    for result in generation_results
    if not result["expected_answerable"]
]


abstention_phrase = (
    "I don't have enough information"
)


abstention_count = 0


for result in out_of_corpus_results:

    if (
        abstention_phrase.lower()
        in result["answer"].lower()
    ):

        abstention_count += 1


out_of_corpus_count = len(
    out_of_corpus_results
)


if out_of_corpus_count > 0:

    abstention_rate = (
        abstention_count
        / out_of_corpus_count
    )

else:

    abstention_rate = 0.0


# ============================================================
# SAVE GENERATION RESULTS
# ============================================================

results_payload = {

    "embedding_model":
        EMBEDDING_MODEL_NAME,

    "generation_model":
        GENERATION_MODEL_NAME,

    "top_k":
        TOP_K,

    "minimum_retrieval_score":
        MIN_RETRIEVAL_SCORE,

    "number_of_questions":
        len(generation_results),

    "answerable_questions":
        answerable_count,

    "answerable_topic_hit_rate":
        answerable_topic_hit_rate,

    "out_of_corpus_questions":
        out_of_corpus_count,

    "abstention_rate":
        abstention_rate,

    "results":
        generation_results,
}


save_json(
    results_payload,
    RESULTS_FILE
)


print(
    "\n✓ Generation results saved:"
)

print(
    RESULTS_FILE
)


# ============================================================
# SAVE EVALUATION CSV
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
            "top_1_topic",
            "expected_topic_found",
            "answer_length",
        ]
    )

    writer.writeheader()

    writer.writerows(
        evaluation_rows
    )


print(
    "✓ Generation evaluation saved:"
)

print(
    EVALUATION_FILE
)


# ============================================================
# FINAL VALIDATION
# ============================================================

validation = {

    "embedding_count":
        int(len(embeddings)),

    "metadata_count":
        int(len(metadata)),

    "embedding_dimension":
        int(embeddings.shape[1]),

    "generation_model_loaded":
        True,

    "test_questions":
        len(generation_results),

    "answerable_questions":
        answerable_count,

    "answerable_topic_hit_rate":
        answerable_topic_hit_rate,

    "out_of_corpus_questions":
        out_of_corpus_count,

    "abstention_rate":
        abstention_rate,

    "ready_for_rag_pipeline":
        bool(
            len(embeddings)
            == len(metadata)

            and np.isfinite(
                embeddings
            ).all()

            and len(generation_results)
            > 0
        ),
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
    "STEP 38G - RAG ANSWER GENERATION COMPLETED"
)

print(
    "=" * 75
)


print(
    f"\nEmbedding model          : "
    f"{EMBEDDING_MODEL_NAME}"
)

print(
    f"Generation model         : "
    f"{GENERATION_MODEL_NAME}"
)

print(
    f"Test questions           : "
    f"{len(generation_results)}"
)

print(
    f"Answerable questions     : "
    f"{answerable_count}"
)

print(
    f"Topic hit rate           : "
    f"{answerable_topic_hit_rate:.2%}"
)

print(
    f"Out-of-corpus questions  : "
    f"{out_of_corpus_count}"
)

print(
    f"Abstention rate          : "
    f"{abstention_rate:.2%}"
)

print(
    f"Ready for RAG pipeline   : "
    f"{validation['ready_for_rag_pipeline']}"
)


print(
    "\nSaved files:"
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
    "\n✓ Retrieval + generation pipeline is operational."
)

print(
    "✓ Next stage can add a production API/chatbot interface."
)