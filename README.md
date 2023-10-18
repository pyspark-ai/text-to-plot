# text-to-plot
Benchmarking tool for generating plots from textual descriptions.

## Datasets and Test Cases

| Dataset                                 | Test Cases | Easy       | Hard       |
|-----------------------------------------|------------|------------|------------|
| `1962_2006_walmart_store_openings.csv`  | 17         | 12         | 5          |
| `2011_february_aa_flight_paths.csv`     | 14         | 11         | 3          |
| `2011_us_ag_exports.csv`                | 21         | 16         | 5          |
| `US-shooting-incidents.csv`             | 24         | 19         | 5          |
| `titanic.csv`                           | 29         | 23         | 6          |
| `winequality-red.csv`                   | 25         | 20         | 5          |
| `us-cities-top-1k.csv`                  | 24         | 19         | 5          |
| **TOTAL**                               | **154**    | 120        | 34         |

## Scripts Overview

| File            | Description |
|-----------------|-------------|
| `src/prep.py`   | Generates test cases, golden plot codes, and saves golden plots in JSON format. |
| `src/evaluate.py` | Runs test cases and compares the output plots with the golden versions. |
