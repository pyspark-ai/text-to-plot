# text-to-plot
Benchmarking tool for generating plots from textual descriptions.

## Datasets and Test Cases

| Dataset                                 | Test Cases |
|-----------------------------------------|------------|
| `1962_2006_walmart_store_openings.csv`  | 12         |
| `2011_february_aa_flight_paths.csv`     | 11         |
| `2011_us_ag_exports.csv`                | 16         |
| `US-shooting-incidents.csv`             | 19         |
| `titanic.csv`                           | 23         |
| `winequality-red.csv`                   | 20         |
| **TOTAL**                               | **101**    |

## Scripts Overview

| File            | Description |
|-----------------|-------------|
| `src/prep.py`   | Generates test cases, golden plot codes, and saves golden plots in JSON format. |
| `src/evaluate.py` | Runs test cases and compares the output plots with the golden versions. |
| `src/util.py`   | Provides utility functions to support benchmarking tasks. |
