"""
HealthAI - Final Agentic AI Integration Test
STEP 17.13

Purpose:
- Validate the complete Agentic AI pipeline.
- Use the existing router, executor and adapters.
- Test each capability with an appropriate input type.
- Do NOT modify the existing Agentic AI architecture.
"""

import os
import sys
import json
from datetime import datetime

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

AGENT_DIR = os.path.join(
    PROJECT_ROOT,
    "src",
    "17_agentic_ai",
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "agentic_ai",
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "agentic_ai_final_integration_evaluation.json",
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

if AGENT_DIR not in sys.path:
    sys.path.insert(0, AGENT_DIR)


# ============================================================
# IMPORT EXISTING EXECUTOR
# ============================================================

from importlib.util import spec_from_file_location, module_from_spec


def load_executor():
    executor_path = os.path.join(
        AGENT_DIR,
        "03_agent_executor.py",
    )

    if not os.path.exists(executor_path):
        raise FileNotFoundError(
            f"Executor not found: {executor_path}"
        )

    spec = spec_from_file_location(
        "healthai_final_executor",
        executor_path,
    )

    module = module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


# ============================================================
# TEST CASES
# ============================================================

DIABETES_PATIENT = {
    "Age": 52,
    "Blood_Glucose_mg_dL": 165,
    "HbA1c_%": 7.8,
    "Total_Cholesterol_mg_dL": 220,
    "BMI": 29.5,
    "Gender": "Male",
    "Region": "Tamil Nadu",
    "Socioeconomic_Status": "Middle",
    "Symptoms": "frequent urination, excessive thirst",
    "BMI_Category": "Overweight",
    "Age_Group": "Middle Age",
    "High_Glucose": 1,
    "High_Cholesterol": 1,
    "HbA1c_Category": "High",
}


LOS_PATIENT = {
    "rcount": 3,
    "gender": "M",
    "dialysisrenalendstage": 0,
    "asthma": 0,
    "irondef": 0,
    "pneum": 0,
    "substancedependence": 0,
    "psychologicaldisordermajor": 0,
    "depress": 0,
    "psychother": 0,
    "fibrosisandother": 0,
    "malnutrition": 0,
    "hemo": 0,
    "hematocrit": 40.0,
    "neutrophils": 65.0,
    "sodium": 139.0,
    "glucose": 110.0,
    "bloodureanitro": 15.0,
    "creatinine": 1.0,
    "bmi": 25.0,
    "pulse": 78.0,
    "respiration": 18.0,
    "secondarydiagnosisnonicd9": 1,
    "facid": "A",
}


TEST_CASES = [
    {
        "id": "AGENT-01",
        "name": "Diabetes prediction",
        "query": "Can you predict my diabetes risk?",
        "expected_tool": "diabetes_prediction",
        "kwargs": {
            "patient_data": DIABETES_PATIENT,
        },
    },

    {
        "id": "AGENT-02",
        "name": "Hospital length of stay",
        "query": (
            "How many days might this patient "
            "stay in the hospital?"
        ),
        "expected_tool": "hospital_los_prediction",
        "kwargs": {
            "patient_data": LOS_PATIENT,
        },
    },

    {
        "id": "AGENT-03",
        "name": "Patient clustering",
        "query": "Can you group similar patients?",
        "expected_tool": "patient_clustering",
        "kwargs": {},
    },

    {
        "id": "AGENT-04",
        "name": "Association analysis",
        "query": (
            "What symptoms and treatments "
            "commonly occur together?"
        ),
        "expected_tool": "association_analysis",
        "kwargs": {},
    },

    {
        "id": "AGENT-05",
        "name": "Chest X-ray analysis",
        "query": (
            "Analyze this chest X-ray "
            "for pneumonia."
        ),
        "expected_tool": "xray_analysis",
        "kwargs": {},
    },

    {
        "id": "AGENT-06",
        "name": "Deterioration prediction",
        "query": (
            "Could this patient deteriorate "
            "in the next few hours?"
        ),
        "expected_tool": "deterioration_prediction",
        "kwargs": {},
    },

    {
        "id": "AGENT-07",
        "name": "Medical NER",
        "query": (
            "Extract the medical conditions and "
            "drugs from this note. Patient has "
            "critical limb ischaemia with stump "
            "pain. Started metformin 500mg bd."
        ),
        "expected_tool": "medical_ner",
        "kwargs": {},
    },

    {
        "id": "AGENT-08",
        "name": "Sentiment analysis",
        "query": (
            "Is this patient feedback "
            "positive or negative?"
        ),
        "expected_tool": "sentiment_analysis",
        "kwargs": {},
    },

    {
        "id": "AGENT-09",
        "name": "Medical RAG",
        "query": "What are the symptoms of diabetes?",
        "expected_tool": "medical_rag",
        "kwargs": {},
    },

    {
        "id": "AGENT-10",
        "name": "Safe fallback",
        "query": "What is the capital of France?",
        "expected_tool": "safe_fallback",
        "kwargs": {},
    },
]


# ============================================================
# RESULT HELPERS
# ============================================================

def get_tool(result):
    return result.get("route")


def get_status(result):
    return result.get("status")


def run_test(executor, test):
    print("\n" + "-" * 70)
    print(f"{test['id']}: {test['name']}")
    print(f"Query: {test['query']}")
    print(f"Expected tool: {test['expected_tool']}")

    try:
        result = executor.run_agent(
            query=test["query"],
            **test["kwargs"],
        )

        predicted_tool = get_tool(result)
        status = get_status(result)

        route_correct = (
            predicted_tool == test["expected_tool"]
        )

        execution_ok = (
            status in {
                "success",
                "fallback",
            }
        )

        passed = (
            route_correct
            and execution_ok
        )

        print(f"Predicted tool: {predicted_tool}")
        print(f"Status: {status}")
        print(
            f"Route correct: "
            f"{'YES' if route_correct else 'NO'}"
        )
        print(
            f"Execution valid: "
            f"{'YES' if execution_ok else 'NO'}"
        )
        print(
            f"Result: "
            f"{'PASS' if passed else 'CHECK'}"
        )

        return {
            "id": test["id"],
            "name": test["name"],
            "query": test["query"],
            "expected_tool": test["expected_tool"],
            "predicted_tool": predicted_tool,
            "status": status,
            "route_correct": route_correct,
            "execution_valid": execution_ok,
            "passed": passed,
        }

    except Exception as exc:

        print(
            f"Execution exception: "
            f"{type(exc).__name__}: {exc}"
        )

        return {
            "id": test["id"],
            "name": test["name"],
            "query": test["query"],
            "expected_tool": test["expected_tool"],
            "predicted_tool": None,
            "status": "error",
            "route_correct": False,
            "execution_valid": False,
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("HEALTHAI - FINAL AGENTIC AI INTEGRATION")
    print("STEP 17.13")
    print("=" * 70)

    print("\nLoading existing Agent Executor...")

    try:
        executor = load_executor()
        print("Agent Executor: AVAILABLE")
    except Exception as exc:
        print(f"Executor loading failed: {exc}")
        return False

    print("\nRunning end-to-end agent tests...")

    results = []

    for test in TEST_CASES:
        result = run_test(
            executor,
            test,
        )
        results.append(result)

    total = len(results)

    route_correct = sum(
        item["route_correct"]
        for item in results
    )

    execution_valid = sum(
        item["execution_valid"]
        for item in results
    )

    passed = sum(
        item["passed"]
        for item in results
    )

    routing_accuracy = (
        route_correct / total
        if total
        else 0
    )

    execution_rate = (
        execution_valid / total
        if total
        else 0
    )

    pass_rate = (
        passed / total
        if total
        else 0
    )

    evaluation = {
        "step": "17.13",
        "component": "Final Agentic AI Integration",
        "timestamp": datetime.now().isoformat(),
        "total_tests": total,
        "correct_routes": route_correct,
        "routing_accuracy": routing_accuracy,
        "valid_executions": execution_valid,
        "execution_success_rate": execution_rate,
        "passed_tests": passed,
        "overall_pass_rate": pass_rate,
        "results": results,
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            evaluation,
            file,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print("\n" + "=" * 70)
    print("FINAL AGENTIC AI SUMMARY")
    print("=" * 70)

    print(f"Total tests          : {total}")
    print(f"Correct routes       : {route_correct}")
    print(
        f"Routing accuracy     : "
        f"{routing_accuracy:.2%}"
    )
    print(f"Valid executions     : {execution_valid}")
    print(
        f"Execution success    : "
        f"{execution_rate:.2%}"
    )
    print(f"Passed tests         : {passed}")
    print(
        f"Overall pass rate    : "
        f"{pass_rate:.2%}"
    )

    print(f"\nEvaluation saved to:")
    print(OUTPUT_PATH)

    if passed == total:

        print(
            "\nSTEP 17.13 "
            "FINAL AGENTIC AI INTEGRATION PASSED"
        )

        return True

    print(
        "\nSTEP 17.13 "
        "FINAL AGENTIC AI INTEGRATION COMPLETED "
        "WITH CHECKS"
    )

    return False


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    success = main()

    if not success:
        raise SystemExit(1)