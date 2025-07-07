# Hybrid Flexible Flow-shop Optimization Thesis

This repository contains all algorithms, implementations, experiments, and analysis from my thesis, which aims to compare multiple metaheuristics and approaches to the Hybrid Flexible Flow-shop (HFFS) problem, a popular production scheduling problem. For a comprehensive overview of the research, methodology, and results, please refer to the [thesis document](final_thesis.pdf).

## Running Experiments

- **Genetic Algorithm Experiments:**
  - Execute `src/ga_experiments.py` to run new experiments for the genetic algorithm.
- **ALNS Experiments:**
  - Execute `src/alns_experiments.py` to run new experiments for the ALNS (Adaptive Large Neighborhood Search) algorithm.
- Both scripts utilize problem instances located in `src/input/`.

## Analysis and Replication

- The `analysis/` directory contains complete analysis notebooks for the results.
- To replicate the results of the thesis, execute the following notebooks:
  - `analysis/alns_analysis.ipynb`: analysis of the ALNS execution results
  - `analysis/ga_analysis.ipynb`: analysis of the GA execution results
  - `analysis/optuna_analysis.ipynb`: analysis of the parameter tuning results of the GA

## Instance Generation

- The `instance_generation/` directory holds all scripts used for the generation and sampling of problem instances.

---

For further details, please consult the thesis document included in this repository.
