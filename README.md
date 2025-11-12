# Predictive Maintenance on NASA C-MAPSS (Turbofan RUL) Dataset Overview

## Background

6. Turbofan Engine Degradation Simulation
Engine degradation simulation was carried out using the Commercial Modular Aero-Propulsion System Simulation (C-MAPSS). Four different sets were simulated under different combinations of operational conditions and fault modes. This records several sensor channels to characterize fault evolution. The data set was provided by the NASA Ames Prognostics Center of Excellence (PCoE).

Download the data: https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip

Data Set Citation: A. Saxena and K. Goebel (2008). ???Turbofan Engine Degradation Simulation Data Set???, NASA Prognostics Data Repository, NASA Ames Research Center, Moffett Field, CA


This project implements a Predictive Maintenance pipeline using the NASA C-MAPSS Turbofan Engine Degradation Dataset  a benchmark for Remaining Useful Life (RUL) estimation.
The goal is to predict when an engine is likely to fail, using multivariate time-series data from engine sensors.

Youll build this project step by step  from raw data exploration to feature engineering, model training, and dashboard visualization  mirroring how predictive maintenance systems are built in industry.

## Objectives

- Learn to work with time-series sensor data (messy, noisy, incomplete).

- Engineer meaningful features to capture degradation trends.

- Train models (LSTM / XGBoost / tree ensembles) to predict RUL.

- Communicate results clearly using visualizations and dashboards.

- Build a portfolio project that demonstrates practical ML + data-engineering skill.

## Dataset Summary

NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set

Each FD subset represents a different experimental setup:

Dataset	Conditions	Fault Modes	Train Engines	Test Engines	Description
FD001	1	1	100	100	HPC Degradation
FD002	6	1	260	259	HPC Degradation (multi-condition)
FD003	1	2	100	100	HPC + Fan Degradation
FD004	6	2	248	249	HPC + Fan Degradation (multi-condition)

Each file (train/test/RUL) contains:

26 space-separated numeric columns

engine_id, cycle, 3 operational settings, and 21 sensors (s1s21)

Test RUL file provides the true RUL for each engines final recorded cycle.

 Project Structure
predictive-maintenance-cmapss/
 data/
    raw/              # Original NASA data files
    interim/          # Intermediate cleaned data
    processed/        # Feature-engineered datasets ready for modeling

 src/
    utils/
        cmapss.py     # Data loading, validation, and helper functions

 notebooks/
    01_data_overview.ipynb   # Step-1: EDA and understanding sensor behavior
    02_preprocessing.ipynb   # Step-2: Labeling, cleaning, and scaling

 dashboard/             # (Later) interactive dashboard or Streamlit app
 requirements.txt
 README.md

 Workflow Overview
Step	Notebook	Description
1	01_data_overview.ipynb	Load raw data, visualize sensor trends, inspect correlations, and note which sensors are meaningful.
2	02_preprocessing.ipynb	Add RUL labels, drop uninformative sensors, scale/normalize features, and save clean datasets.
3	03_modeling.ipynb (future)	Build predictive models (LSTM, XGBoost, etc.) and evaluate RUL predictions.
4	04_dashboard.ipynb (future)	Visualize degradation curves, predicted vs actual RUL, and maintenance schedule simulation.
 Step-1 Recap (Completed)

Youve already:

Implemented src/utils/cmapss.py to robustly load any FD dataset (flat or nested layout).

Loaded FD001 successfully.

Explored:

Sensor behavior vs cycle count

Engine-to-engine variability

Correlations with cycle and RUL

Identified candidate sensors to keep/drop for modeling.

 Step-2 Preview  Labeling & Preprocessing

Next you will:

Add RUL labels to each training row (RUL = max_cycle - cycle).

Drop flat/noisy sensors based on your EDA.

Scale features (per-sensor normalization or standardization).

Save cleaned data as:

data/processed/train_FD001.csv
data/processed/test_FD001.csv


Optionally, visualize distributions post-scaling.

## Setup Instructions
# Clone repo and enter project
git clone https://github.com/<yourname>/predictive-maintenance-cmapss.git
cd predictive-maintenance-cmapss

### Create and activate environment
conda create -n cmapss python=3.11 -y
conda activate cmapss

### Install dependencies
pip install -r requirements.txt

### Start Jupyter
jupyter notebook

 Key Dependencies
Library	Purpose
pandas / numpy	Data manipulation
matplotlib	Visualization
scikit-learn	Preprocessing, scaling, metrics
tensorflow / pytorch	(later) Deep learning models
xgboost / lightgbm	Gradient boosting models
 Reference

A. Saxena, K. Goebel, D. Simon, N. Eklund,
Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,
Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver, CO, 2008.

 Roadmap (high-level)

 Step-1: Data Exploration

 Step-2: Labeling & Preprocessing

 Step-3: Modeling (LSTM, XGBoost)

 Step-4: Dashboard & Reporting

## Why This Project Matters

Predictive maintenance bridges AI and real-world reliability.

This project gives you hands-on experience with:

- Multivariate time-series modeling,
- Feature engineering for physical systems,
- End-to-end ML workflow (data  model  visualization).
- Perfect for a portfolio or interview project in ML Ops / Data Science / IoT AI.
