"""
HealthAI - Final Agent Executor
STEP 17.12

Flow:
User Query -> Registry -> Router -> Selected Tool -> Adapter -> Response

This version uses the actual adapter filenames present in the project.
"""

from pathlib import Path
import importlib.util
import json
from datetime import datetime


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AGENT_DIR = PROJECT_ROOT / "src" / "17_agentic_ai"
REGISTRY_PATH = (
    PROJECT_ROOT / "data" / "processed" / "agentic_ai" / "agent_tool_registry.json"
)
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "agentic_ai"
EVALUATION_PATH = OUTPUT_DIR / "agent_executor_evaluation.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


TOOL_MODULES = {
    "diabetes_prediction": (
        "07_diabetes_prediction_adapter.py",
        "execute_diabetes_prediction",
    ),
    "hospital_los_prediction": (
        "08_hospital_los_adapter.py",
        "execute_hospital_los_prediction",
    ),
    "patient_clustering": (
        "12_patient_clustering_adapter.py",
        "execute_patient_clustering",
    ),
    "association_analysis": (
        "13_association_analysis_adapter.py",
        "execute_association_analysis",
    ),
    "xray_analysis": (
        "14_xray_analysis_adapter.py",
        "execute_xray_analysis",
    ),
    "deterioration_prediction": (
        "15_deterioration_prediction_adapter.py",
        "execute_deterioration_prediction",
    ),
    "medical_ner": (
        "04C_medical_ner_adapter.py",
        "execute_medical_ner",
    ),
    "sentiment_analysis": (
        "16_sentiment_analysis_adapter.py",
        "execute_sentiment_analysis",
    ),
    "medical_translation": (
        "20_medical_translation_adapter.py",
        "execute_medical_translation",
    ),
    "medical_rag": (
        "17_medical_rag_adapter.py",
        "execute_medical_rag",
    ),
    "primary_diagnosis": (
        "05_primary_diagnosis_adapter.py",
        "execute_primary_diagnosis",
    ),
}


def load_module_from_file(filename):
    path = AGENT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Adapter module not found: {path}")

    module_name = (
        "healthai_agent_"
        + filename.replace(".py", "").replace("-", "_")
    )

    spec = importlib.util.spec_from_file_location(module_name, path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module: {filename}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_registry():
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Agent tool registry not found: {REGISTRY_PATH}"
        )

    with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def call_router(query, registry):
    router = load_module_from_file("02_agent_router.py")

    if hasattr(router, "route_query"):
        return router.route_query(query, registry)

    if hasattr(router, "route"):
        try:
            return router.route(query, registry)
        except TypeError:
            return router.route(query)

    raise AttributeError(
        "02_agent_router.py does not expose a supported routing function."
    )


def extract_selected_tool(router_output):
    if isinstance(router_output, str):
        return router_output.strip()

    if not isinstance(router_output, dict):
        return None

    for key in (
        "tool",
        "selected_tool",
        "route",
        "selected_route",
        "destination",
    ):
        value = router_output.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return None


def execute_safe_fallback(query="", **kwargs):
    return {
        "status": "fallback",
        "tool": "safe_fallback",
        "message": (
            "I could not identify a supported HealthAI healthcare task "
            "for this request."
        ),
    }


def _call_function(function, query, kwargs):
    """
    Try the project's common adapter interfaces without masking real
    execution errors.
    """
    attempts = [
        lambda: function(query=query, **kwargs),
        lambda: function(query, **kwargs),
        lambda: function(**kwargs),
    ]

    last_type_error = None

    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_type_error = exc

    if last_type_error:
        raise last_type_error

    raise RuntimeError("Adapter execution failed.")


def execute_tool(tool_name, query="", **kwargs):
    if tool_name == "safe_fallback":
        return execute_safe_fallback(query=query, **kwargs)

    if tool_name not in TOOL_MODULES:
        return {
            "status": "error",
            "tool": tool_name,
            "message": f"Unsupported tool: {tool_name}",
        }

    filename, function_name = TOOL_MODULES[tool_name]

    try:
        module = load_module_from_file(filename)
        function = getattr(module, function_name, None)

        if not callable(function):
            return {
                "status": "error",
                "tool": tool_name,
                "message": (
                    f"Function '{function_name}' was not found "
                    f"in {filename}."
                ),
            }

        result = _call_function(function, query, kwargs)

        if isinstance(result, dict):
            return result

        return {
            "status": "success",
            "tool": tool_name,
            "result": result,
        }

    except Exception as exc:
        return {
            "status": "error",
            "tool": tool_name,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def run_agent(query="", **kwargs):
    if not isinstance(query, str) or not query.strip():
        return {
            "status": "fallback",
            "query": query,
            "route": "safe_fallback",
            "tool": "safe_fallback",
            "message": "Please provide a healthcare-related query.",
        }

    query = query.strip()

    try:
        registry = load_registry()
        router_output = call_router(query, registry)
        selected_tool = extract_selected_tool(router_output)

        if selected_tool is None:
            return {
                "status": "routing_error",
                "query": query,
                "message": "The router did not select a supported tool.",
                "router_output": router_output,
            }

        execution_result = execute_tool(
            selected_tool,
            query=query,
            **kwargs,
        )

        result = {
            "status": execution_result.get("status", "unknown"),
            "query": query,
            "route": selected_tool,
            "tool": selected_tool,
            "answer": execution_result.get("answer"),
            "message": execution_result.get("message"),
            "execution_result": execution_result,
            "router_output": router_output,
            "timestamp": datetime.now().isoformat(),
        }

        # If an adapter returns an explicit answer, promote it to the
        # unified top-level field used by the Streamlit chat.
        if result["answer"] is None and isinstance(
            execution_result.get("result"), str
        ):
            result["answer"] = execution_result["result"]

        return result

    except Exception as exc:
        return {
            "status": "error",
            "query": query,
            "route": "safe_fallback",
            "tool": "safe_fallback",
            "message": "HealthAI agent execution failed.",
            "error": str(exc),
            "timestamp": datetime.now().isoformat(),
        }


def execute_agent_query(query="", **kwargs):
    return run_agent(query=query, **kwargs)


def _test():
    tests = [
        ("What are the symptoms of diabetes?", "medical_rag", {}),
        (
            "Is this patient feedback positive or negative?",
            "sentiment_analysis",
            {},
        ),
        (
            "How many days might this patient stay in the hospital?",
            "hospital_los_prediction",
            {
                "patient_data": {
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
            },
        ),
    ]

    results = []

    for name, (query, expected, kwargs) in enumerate(tests, start=1):
        result = run_agent(query, **kwargs)
        actual = result.get("route")
        passed = actual == expected and result.get("status") not in {
            "error",
            "routing_error",
        }

        results.append(
            {
                "test": name,
                "query": query,
                "expected": expected,
                "actual": actual,
                "status": result.get("status"),
                "passed": passed,
            }
        )

        print(f"TEST {name}: {'PASS' if passed else 'FAIL'}")
        print(f"  Expected: {expected}")
        print(f"  Actual  : {actual}")
        print(f"  Status  : {result.get('status')}")

        if result.get("answer"):
            print(f"  Answer  : {result['answer']}")

    with open(EVALUATION_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    passed = sum(r["passed"] for r in results)
    print(f"\nAgent tests: {passed}/{len(results)}")


if __name__ == "__main__":
    print("=" * 70)
    print("HEALTHAI - FINAL AGENT EXECUTOR")
    print("=" * 70)
    _test()
