# Analytical Card — Patient Clustering

## Method
- Method: K-means-style patient clustering workflow
- Selected K: 2
- Selection basis: Highest recorded Silhouette Score

## Intended use
Group patient records into broad similarity-based profiles for exploratory analysis.

## Evaluation
| K | Silhouette Score |
|---:|---:|
| 2 | 0.1544 |
| 3 | 0.1013 |
| 4 | 0.0924 |
| 5 | 0.0886 |
| 6 | 0.0794 |
| 7 | 0.0765 |
| 8 | 0.0709 |
| 9 | 0.0701 |
| 10 | 0.0671 |

## Final cluster profiles
- Cluster 0: 25,546 patients (25.546%); average age 44.656; glucose 186.03; HbA1c 8.4967; cholesterol 208.50; BMI 24.6038; Type 2 diabetes 91.529%.
- Cluster 1: 74,454 patients (74.454%); average age 44.382; glucose 107.684; HbA1c 5.6766; cholesterol 218.764; BMI 24.5422; Type 2 diabetes 6.4899%.

## Interpretation boundary
These profiles describe observed differences in the development dataset. They do not establish causality, independent clinical diagnosis, or treatment recommendations.

## Data provenance
Indian Healthcare Patient Records — Kaggle (Arun's Workspace), as documented in the project README.
