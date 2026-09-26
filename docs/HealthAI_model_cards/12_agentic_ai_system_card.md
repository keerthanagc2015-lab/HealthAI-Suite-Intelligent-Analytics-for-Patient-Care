# System Card — Agentic AI

## System
Agent router + executor + healthcare-tool adapters.

## Intended use
Route natural-language healthcare requests to the appropriate specialized HealthAI capability.

## Router evaluation
- 14 tests
- 14 correct routes
- Routing accuracy: 100%

## Final integration evaluation
- 10 total tests
- 10 correct routes
- 9 valid executions
- Execution success rate: 90%
- 9 passed tests
- Overall pass rate: 90%

## Failure case
One chest X-ray scenario was routed correctly but produced an execution error. This is documented as an execution-level issue rather than a routing error.

## Safety
The system includes safe fallback handling for unsupported or empty requests.

## Limitations
Routing and execution results reflect the recorded test scenarios and do not establish reliability for all possible healthcare queries.
