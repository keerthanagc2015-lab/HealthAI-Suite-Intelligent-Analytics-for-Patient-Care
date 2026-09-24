"""
HEALTHAI - RAG SOURCE REGISTRY

Step 38A

Defines the authoritative healthcare sources that will form
the initial RAG knowledge base.

Sources:
- WHO
- WHO India
- MedlinePlus / U.S. National Library of Medicine

The registry stores metadata and URLs.
Actual documents will be downloaded in the next step.
"""

import json
import os


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "data/processed/rag/documents"

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "rag_source_registry.json"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# HEALTHCARE SOURCES
# ============================================================

sources = [

    {
        "source_id": "who_diabetes",
        "organization": "World Health Organization",
        "country": "Global",
        "topic": "Diabetes",
        "source_type": "health_topic",
        "url": "https://www.who.int/health-topics/diabetes",
        "authority": "WHO",
    },

    {
        "source_id": "who_india_diabetes",
        "organization": "World Health Organization",
        "country": "India",
        "topic": "Diabetes",
        "source_type": "health_topic",
        "url": "https://www.who.int/india/health-topics/diabetes",
        "authority": "WHO India",
    },

    {
        "source_id": "who_hypertension",
        "organization": "World Health Organization",
        "country": "Global",
        "topic": "Hypertension",
        "source_type": "health_topic",
        "url": "https://www.who.int/health-topics/hypertension/",
        "authority": "WHO",
    },

    {
        "source_id": "who_india_hypertension",
        "organization": "World Health Organization",
        "country": "India",
        "topic": "Hypertension",
        "source_type": "health_topic",
        "url": "https://www.who.int/india/health-topics/hypertension",
        "authority": "WHO India",
    },

    {
        "source_id": "who_ncd",
        "organization": "World Health Organization",
        "country": "Global",
        "topic": "Noncommunicable Diseases",
        "source_type": "fact_sheet",
        "url": (
            "https://www.who.int/news-room/fact-sheets/"
            "detail/noncommunicable-diseases"
        ),
        "authority": "WHO",
    },

    {
        "source_id": "medlineplus_health_topics",
        "organization": "U.S. National Library of Medicine",
        "country": "Global",
        "topic": "General Health",
        "source_type": "health_topic_library",
        "url": "https://medlineplus.gov/healthtopics.html",
        "authority": "MedlinePlus",
    },

    {
        "source_id": "medlineplus_diabetes_selfcare",
        "organization": "U.S. National Library of Medicine",
        "country": "Global",
        "topic": "Type 2 Diabetes Self-Care",
        "source_type": "patient_information",
        "url": (
            "https://medlineplus.gov/ency/"
            "patientinstructions/000328.htm"
        ),
        "authority": "MedlinePlus",
    },

    {
        "source_id": "medlineplus_chronic_illness",
        "organization": "U.S. National Library of Medicine",
        "country": "Global",
        "topic": "Living With Chronic Illness",
        "source_type": "patient_information",
        "url": (
            "https://medlineplus.gov/ency/"
            "patientinstructions/000602.htm"
        ),
        "authority": "MedlinePlus",
    },

]


# ============================================================
# SAVE REGISTRY
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        sources,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# DISPLAY
# ============================================================

print()
print("=" * 75)
print("HEALTHAI - RAG SOURCE REGISTRY")
print("=" * 75)

print()

print(
    f"Registered sources: {len(sources)}"
)

print()

for source in sources:

    print(
        f"[{source['source_id']}] "
        f"{source['organization']} - "
        f"{source['topic']}"
    )

    print(
        f"  Country: {source['country']}"
    )

    print(
        f"  Type: {source['source_type']}"
    )

    print(
        f"  URL: {source['url']}"
    )

    print()


print("=" * 75)

print(
    "✓ Source registry saved:"
)

print(
    os.path.abspath(OUTPUT_FILE)
)

print("=" * 75)