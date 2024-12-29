# Submission for PASS Data & Plotting

This repository contains plotting and data for the PASS submission. The code generates visualizations of experimental data related to fly decision-making behavior and hardware neuron dynamics.

## Overview

The plotting code uses Python with matplotlib, seaborn, and pandas to create publication-quality figures including:

- Trajectory plots showing fly decision-making behavior under different conditions
- Heatmap overlays comparing model predictions to experimental data  
- Autocorrelation analysis of hardware neuron dynamics
- Statistical curve fitting and parameter extraction

## Dependencies

- Python 3.9+
- matplotlib
- seaborn 
- numpy
- pandas
- numba
- pickle
- Eigen 3.4.0

## Directory Structure

The repository is organized as follows:

- `plotting.ipynb`: Main plotting and analysis code
- `modeling/`: Contains Python implementation of behavioral models
  - `Hamerly2019_Data/`: Contains data from Hamerly et al. 2019
  - Includes behavioral simulation of stochastic vs deterministic neuron model
  - Contains model parameter optimization code
- `cpu_benchmark/`: Performance testing and optimization, and power analysis of CPU
  - `pass.cpp`: C++ implementation of CPU benchmark
  - `time_series/`: Contains log files of CPU performance testing, including power analysis from AMDuProf on AMD EPYC 7443P
- `data/`: Contains experimental data
  - Includes pickle files of experimental data
  - Includes solution files for MaxCUT problems


## Usage

The main plotting code is contained in `plotting.ipynb`. Key functionality includes:

- Loading and processing experimental data from pickle files
- Generating multi-panel figures
- Performing autocorrelation analysis
- Fitting exponential decay curves
- Saving publication-ready PDF figures

## Contact

For questions about the code or data, please contact:

Saavan Patel  
saavan@berkeley.edu

## License

This code includes components from the Eigen library which is licensed under the Mozilla Public License v2.0 (see relevant license text in code blocks below):

