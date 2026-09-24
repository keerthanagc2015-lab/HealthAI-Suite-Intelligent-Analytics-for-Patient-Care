"""
HEALTHAI - RAG DOCUMENT INGESTION

Step 38B

Reads the approved RAG source registry, downloads each source,
extracts readable webpage text, cleans it, and saves normalized
documents with metadata.

No embeddings or vector database are created in this step.
"""

import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup


# ============================================================
# CONFIGURATION
# ============================================================

REGISTRY_PATH = (
    "data/processed/rag/documents/"
    "rag_source_registry.json"
)

OUTPUT_DIR = (
    "data/processed/rag/documents/"
    "normalized"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/151.0 Safari/537.36"
    )
}


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 75)
print("HEALTHAI - RAG DOCUMENT INGESTION")
print("=" * 75)


# ============================================================
# LOAD SOURCE REGISTRY
# ============================================================

if not os.path.exists(REGISTRY_PATH):

    raise FileNotFoundError(
        f"""
Source registry not found:

{os.path.abspath(REGISTRY_PATH)}

Run Step 38A first.
"""
    )


with open(
    REGISTRY_PATH,
    "r",
    encoding="utf-8"
) as f:

    sources = json.load(f)


print()
print(
    f"✓ Loaded {len(sources)} registered sources."
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Normalize extracted webpage text.
    """

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Remove leading/trailing whitespace
    text = text.strip()

    return text


# ============================================================
# EXTRACT WEBPAGE TEXT
# ============================================================

def extract_webpage_text(html):

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # Remove elements that normally contain
    # navigation, scripts, styling or irrelevant content.
    for element in soup([
        "script",
        "style",
        "noscript",
        "svg",
        "nav",
        "footer",
        "header"
    ]):

        element.decompose()

    # Prefer article/main content when available.
    main = soup.find("main")

    if main is not None:

        text = main.get_text(
            separator=" ",
            strip=True
        )

    else:

        article = soup.find("article")

        if article is not None:

            text = article.get_text(
                separator=" ",
                strip=True
            )

        else:

            text = soup.get_text(
                separator=" ",
                strip=True
            )

    return clean_text(text)


# ============================================================
# DOWNLOAD + PROCESS
# ============================================================

successful = []
failed = []


for index, source in enumerate(
    sources,
    start=1
):

    source_id = source["source_id"]
    url = source["url"]

    print()
    print("-" * 75)

    print(
        f"[{index}/{len(sources)}] "
        f"{source_id}"
    )

    print(
        f"Topic: {source['topic']}"
    )

    print(
        f"URL: {url}"
    )

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            ""
        )

        if (
            "text/html"
            not in content_type.lower()
        ):

            raise ValueError(
                f"Expected HTML but received: "
                f"{content_type}"
            )

        text = extract_webpage_text(
            response.text
        )

        if len(text) < 500:

            raise ValueError(
                f"Extracted text is too short "
                f"({len(text)} characters)"
            )


        # ----------------------------------------------------
        # CREATE NORMALIZED DOCUMENT
        # ----------------------------------------------------

        document = {

            "document_id": source_id,

            "title": (
                f"{source['organization']} - "
                f"{source['topic']}"
            ),

            "text": text,

            "metadata": {

                "source_id": source_id,

                "organization": (
                    source["organization"]
                ),

                "country": (
                    source["country"]
                ),

                "topic": (
                    source["topic"]
                ),

                "source_type": (
                    source["source_type"]
                ),

                "authority": (
                    source["authority"]
                ),

                "source_url": url,

                "content_type": content_type,

                "ingestion_status": "success"
            }
        }


        output_path = os.path.join(
            OUTPUT_DIR,
            f"{source_id}.json"
        )


        with open(
            output_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                document,
                f,
                indent=4,
                ensure_ascii=False
            )


        successful.append(
            source_id
        )


        print(
            f"✓ Downloaded successfully"
        )

        print(
            f"  Characters: {len(text):,}"
        )

        print(
            f"  Saved: {output_path}"
        )


    except Exception as error:

        failed.append(
            {
                "source_id": source_id,
                "url": url,
                "error": str(error)
            }
        )

        print(
            f"✗ Failed: {error}"
        )


    # Small delay between requests
    time.sleep(1)


# ============================================================
# SAVE INGESTION REPORT
# ============================================================

report = {

    "total_sources": len(sources),

    "successful_sources": len(
        successful
    ),

    "failed_sources": len(
        failed
    ),

    "successful_source_ids": successful,

    "failed_sources": failed,

    "output_directory": os.path.abspath(
        OUTPUT_DIR
    )
}


REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "rag_ingestion_report.json"
)


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 75)
print("INGESTION SUMMARY")
print("=" * 75)

print()
print(
    f"Total sources     : {len(sources)}"
)

print(
    f"Successful        : {len(successful)}"
)

print(
    f"Failed            : {len(failed)}"
)


if failed:

    print()
    print("Failed sources:")

    for item in failed:

        print(
            f"  - {item['source_id']}: "
            f"{item['error']}"
        )


print()
print("✓ Ingestion report saved:")

print(
    os.path.abspath(REPORT_PATH)
)


print()
print("=" * 75)

if len(failed) == 0:

    print(
        "STEP 38B - COMPLETED SUCCESSFULLY"
    )

else:

    print(
        "STEP 38B - COMPLETED WITH FAILURES"
    )

print("=" * 75)

print()
print("Next:")
print("38C - RAG Document Quality Validation")
