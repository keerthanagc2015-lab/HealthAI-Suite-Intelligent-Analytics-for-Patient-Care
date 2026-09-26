# System Card — Medical RAG

## System
Retrieval-augmented medical-information assistant using embeddings, retrieval, reranking and a safety gate.

## Intended use
Provide evidence-grounded healthcare information from a curated source set in the HealthAI portfolio application.

## Retrieval infrastructure
- 55 stored medical vectors
- 384-dimensional embedding space
- 55 metadata records
- 0 duplicate chunk IDs
- 6 retrieval tests
- 6 successful retrieval queries
- Top-3 topic hit rate: 100%

## Final evaluation
| Measure | Result |
|---|---:|
| Top-1 Topic Accuracy | 1.0000 |
| Top-3 Topic Hit Rate | 1.0000 |
| Mean Topic Consistency | 0.8333 |
| In-domain Acceptance | 1.0000 |
| Out-of-domain Rejection | 1.0000 |
| Safety Score | 0.9333 |
| Context Acceptance / Generation / Extraction | 0.8571 |

## Safety design
The pipeline uses retrieval, question-aware reranking and a semantic-similarity safety threshold. When sufficient evidence is unavailable, the system can abstain instead of producing an unsupported medical-information response.

## Sources
Eight curated public sources from WHO, WHO India and MedlinePlus are documented in the project README.

## Limitations
These results describe the recorded evaluation set. They do not establish clinical accuracy or medical safety in real-world populations.
