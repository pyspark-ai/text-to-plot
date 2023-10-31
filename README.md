# Text-to-Plot Benchmark Tool

A tool designed to evaluate the capability of systems to generate plots based on textual descriptions.

## Overview

The **Text-to-Plot** benchmark encompasses datasets from diverse categories such as retail, aviation, agriculture,
public safety, maritime, viticulture, and urban demographics. The benchmark consists of **232** test cases, split
between **149** training cases and **83** testing cases. The test cases are further categorized into easy and hard
levels, with the hard cases constituting 25% of the total. In easy test cases, we specify the plot type, while in
hard test cases, we let the LLM decide.

## Datasets

### Training Datasets

| Train Datasets                          | Test Cases | Easy       | Hard       |
|-----------------------------------------|------------|------------|------------|
| `european_turnout.csv`                  | 12         | 8          | 4          |
| `1962_2006_walmart_store_openings.csv`  | 17         | 12         | 5          |
| `2011_february_aa_flight_paths.csv`     | 14         | 11         | 3          |
| `2011_us_ag_exports.csv`                | 21         | 16         | 5          |
| `US-shooting-incidents.csv`             | 24         | 19         | 5          |
| `winequality-red.csv`                   | 25         | 20         | 5          |
| `us-cities-top-1k.csv`                  | 24         | 19         | 5          |
| `medicare.csv`                          | 12         | 7          | 5          |
| **TOTAL**                               | **149**    | 112        | 37         |

**TOTAL**: 149 (112 Easy, 37 Hard)

### Testing Datasets

| Test Datasets                           | Test Cases | Easy       | Hard       |
|-----------------------------------------|------------|------------|------------|
| `mpg.csv`                               | 13         | 10         | 3          |
| `gapminder2007.csv`                     | 19         | 13         | 6          |
| `volcano_db.csv`                        | 22         | 16         | 6          |
| `titanic.csv`                           | 29         | 23         | 6          |
| **TOTAL**                               | **83**     | 62         | 21         |

**TOTAL**: 83 (62 Easy, 21 Hard)

## Scripts

- **`src/prep.py`**: Generates test cases and saves golden plots in JSON.
- **`src/evaluate.py`**: Evaluates test cases against golden plots.
