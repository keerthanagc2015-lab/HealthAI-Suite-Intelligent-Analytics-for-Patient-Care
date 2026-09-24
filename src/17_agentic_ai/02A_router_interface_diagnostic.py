"""
HealthAI Agentic AI
Router Interface Diagnostic

Purpose:
    Diagnose the exact interface exposed by 02_agent_router.py.

This script does NOT modify the router.
It only:
    1. Imports the router.
    2. Shows available functions.
    3. Calls likely router functions.
    4. Prints the exact returned object.
    5. Prints the exact exception if the call fails.
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

print("=" * 70)
print("HEALTHAI ROUTER INTERFACE DIAGNOSTIC")
print("=" * 70)

print(f"\nProject root:")
print(PROJECT_ROOT)


# ============================================================
# IMPORT ROUTER
# ============================================================

print("\nImporting 02_agent_router.py...")

try:

    router_module = importlib.import_module(
        "02_agent_router"
    )

    print("Router import: SUCCESS")

except Exception as exc:

    print("Router import: FAILED")

    print(
        f"\nException type: {type(exc).__name__}"
    )

    print(
        f"Exception message: {exc}"
    )

    raise SystemExit(1)


# ============================================================
# INSPECT MODULE
# ============================================================

print("\nFunctions/classes exposed by router:")
print("-" * 70)

for name in dir(router_module):

    if name.startswith("_"):
        continue

    obj = getattr(
        router_module,
        name,
    )

    if inspect.isfunction(obj):

        try:
            signature = inspect.signature(obj)
        except Exception:
            signature = "unknown"

        print(
            f"FUNCTION : {name}{signature}"
        )

    elif inspect.isclass(obj):

        print(
            f"CLASS    : {name}"
        )


# ============================================================
# FIND ROUTER FUNCTION
# ============================================================

candidate_functions = [
    "route_query",
    "route",
    "classify_query",
    "predict_route",
    "get_route",
    "route_request",
    "agent_router",
]


print("\nCandidate router functions:")
print("-" * 70)

for function_name in candidate_functions:

    if hasattr(
        router_module,
        function_name,
    ):

        function = getattr(
            router_module,
            function_name,
        )

        if callable(function):

            try:
                signature = inspect.signature(
                    function
                )
            except Exception:
                signature = "unknown"

            print(
                f"FOUND: {function_name}{signature}"
            )


# ============================================================
# TEST QUERIES
# ============================================================

TEST_QUERIES = [

    (
        "Diabetes",
        "Can you predict diabetes risk for this patient?"
    ),

    (
        "Medical NER",
        (
            "Patient has critical limb ischaemia "
            "with stump pain. Started metformin "
            "500mg bd and apixaban."
        )
    ),

    (
        "RAG",
        "What are the symptoms of diabetes?"
    ),

    (
        "Out of domain",
        "What is the capital of France?"
    ),
]


# ============================================================
# CALL FUNCTION
# ============================================================

def test_function(
    function_name: str,
    function,
):

    print("\n" + "=" * 70)

    print(
        f"TESTING FUNCTION: {function_name}"
    )

    print("=" * 70)

    try:

        signature = inspect.signature(
            function
        )

        print(
            f"\nSignature: {signature}"
        )

    except Exception:

        signature = None

    for name, query in TEST_QUERIES:

        print(
            f"\n--- {name} ---"
        )

        print(
            f"Query: {query!r}"
        )

        # ----------------------------------------------------
        # Try positional argument
        # ----------------------------------------------------

        try:

            result = function(
                query
            )

            print(
                "\nPOSITIONAL CALL: SUCCESS"
            )

            print(
                f"Return type: "
                f"{type(result).__name__}"
            )

            print(
                f"Return value:\n"
                f"{result!r}"
            )

            continue

        except Exception as exc:

            print(
                "\nPOSITIONAL CALL: FAILED"
            )

            print(
                f"Exception type: "
                f"{type(exc).__name__}"
            )

            print(
                f"Exception: {exc}"
            )

        # ----------------------------------------------------
        # Try query= keyword
        # ----------------------------------------------------

        try:

            result = function(
                query=query
            )

            print(
                "\nQUERY= CALL: SUCCESS"
            )

            print(
                f"Return type: "
                f"{type(result).__name__}"
            )

            print(
                f"Return value:\n"
                f"{result!r}"
            )

            continue

        except Exception as exc:

            print(
                "\nQUERY= CALL: FAILED"
            )

            print(
                f"Exception type: "
                f"{type(exc).__name__}"
            )

            print(
                f"Exception: {exc}"
            )

        # ----------------------------------------------------
        # Try text= keyword
        # ----------------------------------------------------

        try:

            result = function(
                text=query
            )

            print(
                "\nTEXT= CALL: SUCCESS"
            )

            print(
                f"Return type: "
                f"{type(result).__name__}"
            )

            print(
                f"Return value:\n"
                f"{result!r}"
            )

            continue

        except Exception as exc:

            print(
                "\nTEXT= CALL: FAILED"
            )

            print(
                f"Exception type: "
                f"{type(exc).__name__}"
            )

            print(
                f"Exception: {exc}"
            )


# ============================================================
# TEST CANDIDATES
# ============================================================

found = False

for function_name in candidate_functions:

    if hasattr(
        router_module,
        function_name,
    ):

        function = getattr(
            router_module,
            function_name,
        )

        if callable(function):

            found = True

            test_function(
                function_name,
                function,
            )


# ============================================================
# CLASS-BASED ROUTER
# ============================================================

if hasattr(
    router_module,
    "AgentRouter",
):

    print(
        "\n" + "=" * 70
    )

    print(
        "FOUND AgentRouter CLASS"
    )

    print(
        "=" * 70
    )

    try:

        router = (
            router_module
            .AgentRouter()
        )

        print(
            "AgentRouter instance created."
        )

        if hasattr(
            router,
            "route",
        ):

            test_function(
                "AgentRouter.route",
                router.route,
            )

    except Exception as exc:

        print(
            f"AgentRouter failed: {exc}"
        )


# ============================================================
# FINAL
# ============================================================

print(
    "\n" + "=" * 70
)

print(
    "ROUTER DIAGNOSTIC COMPLETED"
)

print(
    "=" * 70
)

if not found:

    print(
        "\nNo standard router function was found."
    )

    print(
        "We need to inspect the actual 02_agent_router.py."
    )

else:

    print(
        "\nCopy the diagnostic output and send it to me."
    )