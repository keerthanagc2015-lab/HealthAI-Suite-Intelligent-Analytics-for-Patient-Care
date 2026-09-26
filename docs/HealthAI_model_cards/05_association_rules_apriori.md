# Analytical Card — Healthcare Association Rule Mining

## Method
- Method: Apriori-based association rule mining
- Transactions: 100,000
- Unique items: 39
- Minimum support: 0.01
- Minimum patient count: 1,000

## Rule-generation results
- Frequent itemsets: 858
- Initial rules generated: 2,479
- Rules with lift > 1: 1,669
- After support/confidence/lift filtering: 451
- After imaging filter: 61
- After same-feature filtering: 61 final meaningful rules

## Intended use
Identify recurring co-occurrence patterns in the project's healthcare dataset for exploratory analysis.

## Evaluation / interpretation
Rules are evaluated using support, confidence and lift, followed by domain-specific filtering. They describe associations, not causation.

## Data provenance
Indian Healthcare Patient Records — Kaggle (Arun's Workspace), as documented in the project README.

## Limitations
- Association does not imply a clinical causal relationship.
- Rules are dataset-dependent and require domain review before any operational use.
