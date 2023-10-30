# Text-to-Plot Benchmark Tool

A tool designed to evaluate the capability of systems to generate plots based on textual descriptions.

## Overview

The **Text-to-Plot** benchmark encompasses datasets from diverse categories such as retail, aviation, agriculture,
public safety, maritime, viticulture, and urban demographics. The benchmark consists of **213** test cases, split
between **154** training cases and **59** testing cases. The test cases are further categorized into easy and hard
levels, with the testing dataset having a higher percentage of hard cases (~42%) compared to the training dataset (~22%)
.

## Datasets

### Training Datasets

| Train Datasets                          | Test Cases | Easy       | Hard       |
|-----------------------------------------|------------|------------|------------|
| `1962_2006_walmart_store_openings.csv`  | 17         | 12         | 5          |
| `2011_february_aa_flight_paths.csv`     | 14         | 11         | 3          |
| `2011_us_ag_exports.csv`                | 21         | 16         | 5          |
| `US-shooting-incidents.csv`             | 24         | 19         | 5          |
| `titanic.csv`                           | 29         | 23         | 6          |
| `winequality-red.csv`                   | 25         | 20         | 5          |
| `us-cities-top-1k.csv`                  | 24         | 19         | 5          |
| **TOTAL**                               | **154**    | 120        | 34         |

**TOTAL**: 154 (120 Easy, 34 Hard)

### Testing Datasets

| Test Datasets                           | Test Cases | Easy       | Hard       |
|-----------------------------------------|------------|------------|------------|
| `european_turnout.csv`                  | 12         | 8          | 4          |
| `mpg.csv`                               | 13         | 10         | 3          |
| `gapminder2007.csv`                     | 19         | 13         | 6          |
| `medicare.csv`                          | 12         | 7          | 5          |
| `volcano_db.csv`                        | 23         | 16         | 7          |
| **TOTAL**                               | **79**     | 54         | 25         |

**TOTAL**: 59 (34 Easy, 25 Hard)

## Scripts

- **`src/prep.py`**: Generates test cases and saves golden plots in JSON.
- **`src/evaluate.py`**: Evaluates test cases against golden plots.
