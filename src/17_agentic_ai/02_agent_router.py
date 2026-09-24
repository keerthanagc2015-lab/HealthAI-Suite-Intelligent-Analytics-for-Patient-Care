"""
HealthAI Agentic AI
Step 17.2 - Agent Router

Routes healthcare queries to the appropriate HealthAI tool.

Interface used by Agent Executor:

    route_query(query, registry)

The registry provides available tools.
Routing rules are maintained by this router.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REGISTRY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "agentic_ai"
    / "agent_tool_registry.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "agentic_ai"
    / "agent_router_evaluation.json"
)


# ============================================================
# ROUTING RULES
# ============================================================

ROUTING_RULES = {

    "diabetes_prediction": {

        "phrases": [
            "predict diabetes",
            "diabetes risk",
            "diabetes prediction",
            "risk of diabetes",
            "diabetes probability",
        ],

        "keywords": [
            "diabetes",
            "glucose",
            "hba1c",
            "blood sugar",
        ],
    },


    "hospital_los_prediction": {

        "phrases": [
            "length of stay",
            "hospital stay",
            "how many days",
            "how long stay",
            "stay in hospital",
            "stay in the hospital",
        ],

        "keywords": [
            "hospital",
            "stay",
            "admission",
            "length",
        ],
    },


    "patient_clustering": {

        "phrases": [
            "group similar patients",
            "group patients",
            "cluster patients",
            "patient clusters",
            "segment patients",
            "patient segmentation",
        ],

        "keywords": [
            "cluster",
            "clustering",
            "group",
            "segment",
            "similar patients",
        ],
    },


    "association_analysis": {

        "phrases": [
            "occur together",
            "commonly occur together",
            "association between",
            "associated with",
            "frequently occur",
            "common relationships",
        ],

        "keywords": [
            "association",
            "together",
            "relationship",
            "frequent",
            "cooccur",
        ],
    },


    "xray_analysis": {

        "phrases": [
            "chest x-ray",
            "chest xray",
            "x-ray",
            "xray",
            "pneumonia image",
            "pneumonia x-ray",
        ],

        "keywords": [
            "xray",
            "x-ray",
            "chest",
            "image",
            "pneumonia",
        ],
    },


    "deterioration_prediction": {

        "phrases": [
            "patient deteriorate",
            "patient deterioration",
            "deteriorate in the next",
            "deterioration in the next",
            "clinical deterioration",
        ],

        "keywords": [
            "deteriorate",
            "deterioration",
            "sepsis",
            "risk",
            "next few hours",
        ],
    },


    "medical_ner": {

        "phrases": [
            "extract medical conditions",
            "extract medical entities",
            "extract conditions and drugs",
            "extract the medical conditions",
            "extract drugs",
            "medical entities",
            "clinical entities",
            "from this note",
        ],

        "keywords": [
            "extract",
            "medical",
            "condition",
            "conditions",
            "drug",
            "drugs",
            "medication",
            "medications",
            "clinical note",
            "medical note",
        ],
    },


    "sentiment_analysis": {

        "phrases": [
            "patient feedback",
            "hospital feedback",
            "positive or negative",
            "sentiment",
            "patient sentiment",
            "feedback sentiment",
        ],

        "keywords": [
            "feedback",
            "sentiment",
            "positive",
            "negative",
            "satisfaction",
        ],
    },


    "medical_translation": {
        "phrases": [
            "translate this",
            "translate to hindi",
            "translate into hindi",
            "medical translation",
            "translate medical",
            "translate healthcare",
            "translate this medical text"
        ],
        "keywords": [
            "translate",
            "translation",
            "hindi",
            "medical translation",
            "translator"
        ]
    },

    "medical_rag": {

        "phrases": [
            "symptoms of diabetes",
            "manage diabetes",
            "diabetes management",
            "what is diabetes",
            "what is hypertension",
            "high blood pressure",
            "what is high blood pressure",
            "symptoms of hypertension",
            "what is a chronic illness",
            "chronic illness",
            "living with chronic illness",
            "health information",
        ],

        "keywords": [
            "symptoms",
            "hypertension",
            "blood pressure",
            "chronic illness",
            "diabetes",
        ],
    },
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:

    if not isinstance(text, str):

        return ""

    text = text.lower().strip()

    text = re.sub(
        r"[^a-z0-9\s\-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# PHRASE MATCHING
# ============================================================

def find_phrase_matches(
    query: str,
    phrases: list[str]
) -> list[str]:

    matches = []

    for phrase in phrases:

        phrase_normalized = normalize_text(
            phrase
        )

        if phrase_normalized in query:

            matches.append(
                phrase
            )

    return matches


# ============================================================
# TOOL SCORING
# ============================================================

def score_tool(
    query: str,
    tool_id: str
) -> dict[str, Any]:

    rules = ROUTING_RULES.get(
        tool_id
    )

    if not rules:

        return {

            "tool":
                tool_id,

            "score":
                0.0,

            "matched_signals":
                []
        }


    normalized_query = normalize_text(
        query
    )


    matched_signals = []


    # --------------------------------------------------------
    # Phrase matches
    # --------------------------------------------------------

    phrase_matches = find_phrase_matches(
        normalized_query,
        rules.get(
            "phrases",
            []
        )
    )


    for phrase in phrase_matches:

        matched_signals.append(
            f"phrase:{phrase}"
        )


    # --------------------------------------------------------
    # Keyword matches
    # --------------------------------------------------------

    keyword_matches = []

    query_words = set(
        normalized_query.split()
    )


    for keyword in rules.get(
        "keywords",
        []
    ):

        keyword_normalized = normalize_text(
            keyword
        )


        # Multi-word keyword
        if " " in keyword_normalized:

            if keyword_normalized in normalized_query:

                keyword_matches.append(
                    keyword
                )

        else:

            if keyword_normalized in query_words:

                keyword_matches.append(
                    keyword
                )


    for keyword in keyword_matches:

        matched_signals.append(
            f"keyword:{keyword}"
        )


    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    phrase_score = min(
        len(phrase_matches) * 0.40,
        0.80
    )


    keyword_score = min(
        len(keyword_matches) * 0.10,
        0.40
    )


    score = min(
        phrase_score + keyword_score,
        1.0
    )


    return {

        "tool":
            tool_id,

        "score":
            round(
                score,
                4
            ),

        "matched_signals":
            matched_signals,

        "phrase_matches":
            phrase_matches,

        "keyword_matches":
            keyword_matches,
    }


# ============================================================
# ROUTER
# ============================================================

def route_query(
    query: str,
    registry: dict[str, Any]
) -> dict[str, Any]:

    normalized_query = normalize_text(
        query
    )


    # --------------------------------------------------------
    # Empty query
    # --------------------------------------------------------

    if not normalized_query:

        return {

            "query":
                query,

            "normalized_query":
                normalized_query,

            "decision":
                "safe_fallback",

            "selected_tool":
                "safe_fallback",

            "confidence":
                0.0,

            "reason":
                "Empty query.",

            "matched_signals":
                [],

            "tool_scores":
                []
        }


    # --------------------------------------------------------
    # Get registered tools
    # --------------------------------------------------------

    registered_tools = registry.get(
        "tools",
        {}
    )


    if isinstance(
        registered_tools,
        dict
    ):

        available_tools = list(
            registered_tools.keys()
        )

    elif isinstance(
        registered_tools,
        list
    ):

        available_tools = []

        for tool in registered_tools:

            if isinstance(
                tool,
                dict
            ):

                tool_id = (
                    tool.get("tool_id")
                    or tool.get("tool_name")
                    or tool.get("name")
                )

                if tool_id:

                    available_tools.append(
                        tool_id
                    )

    else:

        available_tools = []


    # --------------------------------------------------------
    # Score available tools
    # --------------------------------------------------------

    scores = []

    for tool_id in available_tools:

        if tool_id not in ROUTING_RULES:

            continue

        result = score_tool(
            normalized_query,
            tool_id
        )

        scores.append(
            result
        )


    # --------------------------------------------------------
    # No rules matched
    # --------------------------------------------------------

    if not scores:

        return {

            "query":
                query,

            "normalized_query":
                normalized_query,

            "decision":
                "safe_fallback",

            "selected_tool":
                "safe_fallback",

            "confidence":
                0.0,

            "reason":
                "No routing rules matched the query.",

            "matched_signals":
                [],

            "tool_scores":
                []
        }


    # --------------------------------------------------------
    # Sort by score
    # --------------------------------------------------------

    scores.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    best = scores[0]

    best_score = best[
        "score"
    ]


    # --------------------------------------------------------
    # Safety threshold
    # --------------------------------------------------------

    MIN_ROUTING_CONFIDENCE = 0.25


    if best_score < MIN_ROUTING_CONFIDENCE:

        return {

            "query":
                query,

            "normalized_query":
                normalized_query,

            "decision":
                "safe_fallback",

            "selected_tool":
                "safe_fallback",

            "confidence":
                best_score,

            "reason":
                (
                    "No supported capability "
                    "matched with sufficient confidence."
                ),

            "matched_signals":
                best.get(
                    "matched_signals",
                    []
                ),

            "tool_scores":
                scores
        }


    # --------------------------------------------------------
    # Successful route
    # --------------------------------------------------------

    return {

        "query":
            query,

        "normalized_query":
            normalized_query,

        "decision":
            best["tool"],

        "selected_tool":
            best["tool"],

        "confidence":
            best_score,

        "reason":
            "Matched routing rules.",

        "matched_signals":
            best.get(
                "matched_signals",
                []
            ),

        "tool_scores":
            scores
    }


# ============================================================
# LOAD REGISTRY
# ============================================================

def load_registry() -> dict[str, Any]:

    with open(
        REGISTRY_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(
    results: list[dict[str, Any]]
) -> None:

    print(
        "\n" + "=" * 70
    )

    print(
        "HEALTHAI AGENT ROUTER"
    )

    print(
        "STEP 17.2 - ROUTING EVALUATION"
    )

    print(
        "=" * 70
    )


    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"\nTEST {index}: "
            f"{result['name']}"
        )

        print(
            f"Query     : "
            f"{result['query']}"
        )

        print(
            f"Expected  : "
            f"{result['expected']}"
        )

        print(
            f"Predicted : "
            f"{result['predicted']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence']:.3f}"
        )

        print(
            f"Correct   : "
            f"{result['correct']}"
        )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_router(
    registry: dict[str, Any],
    test_cases: list[dict[str, str]]
) -> list[dict[str, Any]]:

    results = []


    for test_case in test_cases:

        output = route_query(
            test_case["query"],
            registry
        )


        predicted = output.get(
            "selected_tool",
            "safe_fallback"
        )


        results.append({

            "name":
                test_case["name"],

            "query":
                test_case["query"],

            "expected":
                test_case["expected"],

            "predicted":
                predicted,

            "confidence":
                output.get(
                    "confidence",
                    0.0
                ),

            "reason":
                output.get(
                    "reason"
                ),

            "matched_signals":
                output.get(
                    "matched_signals",
                    []
                ),

            "correct":
                predicted
                == test_case["expected"]
        })


    return results


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [

    {
        "name":
            "Diabetes risk",

        "query":
            "Can you predict my diabetes risk?",

        "expected":
            "diabetes_prediction"
    },


    {
        "name":
            "Hospital length of stay",

        "query":
            "How many days might this patient stay in the hospital?",

        "expected":
            "hospital_los_prediction"
    },


    {
        "name":
            "Patient clustering",

        "query":
            "Can you group similar patients?",

        "expected":
            "patient_clustering"
    },


    {
        "name":
            "Association analysis",

        "query":
            "What symptoms and treatments commonly occur together?",

        "expected":
            "association_analysis"
    },


    {
        "name":
            "Chest X-ray",

        "query":
            "Analyze this chest X-ray for pneumonia.",

        "expected":
            "xray_analysis"
    },


    {
        "name":
            "Deterioration",

        "query":
            "Could this patient deteriorate in the next few hours?",

        "expected":
            "deterioration_prediction"
    },


    {
        "name":
            "Medical NER",

        "query":
            (
                "Extract the medical conditions and drugs "
                "from this note. Patient has critical limb "
                "ischaemia with stump pain. Started metformin "
                "500mg bd and apixaban."
            ),

        "expected":
            "medical_ner"
    },


    {
        "name":
            "Medical NER 2",

        "query":
            (
                "Extract the medical conditions and drugs "
                "from this note. Patient has controlled HTN. "
                "BP is 168/102 mmHg with headache and dizziness."
            ),

        "expected":
            "medical_ner"
    },


    {
        "name":
            "Sentiment",

        "query":
            "Is this patient feedback positive or negative?",

        "expected":
            "sentiment_analysis"
    },


    {
        "name":
            "Diabetes knowledge",

        "query":
            "What are the symptoms of diabetes?",

        "expected":
            "medical_rag"
    },


    {
        "name":
            "Hypertension knowledge",

        "query":
            "What is high blood pressure?",

        "expected":
            "medical_rag"
    },


    {
        "name":
            "Chronic illness knowledge",

        "query":
            "What is a chronic illness?",

        "expected":
            "medical_rag"
    },


    {
        "name":
            "Out of domain",

        "query":
            "What is the capital of France?",

        "expected":
            "safe_fallback"
    },


    {
        "name":
            "Empty query",

        "query":
            "",

        "expected":
            "safe_fallback"
    },
]


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results: list[dict[str, Any]]
) -> None:

    correct = sum(
        1
        for result in results
        if result["correct"]
    )


    accuracy = (
        correct / len(results)
        if results
        else 0.0
    )


    output = {

        "step":
            "17.2",

        "component":
            "Agent Router",

        "total_tests":
            len(results),

        "correct_routes":
            correct,

        "routing_accuracy":
            round(
                accuracy,
                4
            ),

        "routing_rules_count":
            len(
                ROUTING_RULES
            ),

        "results":
            results
    }


    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# MAIN
# ============================================================

def main():

    registry = load_registry()


    print(
        f"Loaded registry with "
        f"{registry.get('tool_count', 0)} tools."
    )


    print(
        f"Router contains "
        f"{len(ROUTING_RULES)} routing rule sets."
    )


    results = evaluate_router(
        registry,
        TEST_CASES
    )


    display_results(
        results
    )


    save_results(
        results
    )


    correct = sum(
        result["correct"]
        for result in results
    )


    accuracy = (
        correct / len(results)
        if results
        else 0.0
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "ROUTER SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Total tests      : "
        f"{len(results)}"
    )

    print(
        f"Correct routes   : "
        f"{correct}"
    )

    print(
        f"Routing accuracy : "
        f"{accuracy:.2%}"
    )

    print(
        f"Saved to         : "
        f"{OUTPUT_PATH}"
    )


    if accuracy == 1.0:

        print(
            "\nSTEP 17.2 COMPLETED SUCCESSFULLY"
        )

    else:

        print(
            "\nSTEP 17.2 COMPLETED WITH CHECKS"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()