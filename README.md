# Predictive Maintenance on NASA C-MAPSS Turbofan Dataset

## Overview

This project implements a **Predictive Maintenance** pipeline using the **NASA C-MAPSS Turbofan Engine Degradation Dataset** — a benchmark for Remaining Useful Life (RUL) estimation.  
The goal is to **predict when an engine is likely to fail**, using multivariate time-series data from engine sensors.

You’ll build this project step by step — from raw data exploration to feature engineering, model training, and dashboard visualization — mirroring how predictive maintenance systems are built in industry.

---

## 🎯 Objectives

- Learn to work with **time-series sensor data** (messy, noisy, incomplete).  
- Engineer meaningful features to capture **degradation trends**.  
- Train models (LSTM / XGBoost / tree ensembles) to **predict RUL**.  
- Communicate results clearly using **visualizations and dashboards**.  
- Build a portfolio project that demonstrates practical ML + data-engineering skill.

---

## 🧩 Dataset Summary

**NASA C-MAPSS Turbofan Engine Degradation Simulation Data Set**

Each “FD” subset represents a different experimental setup:

| Dataset | Conditions | Fault Modes | Train Engines | Test Engines | Description |
|---------|------------|-------------|----------------|---------------|-------------|
| **FD001** | 1 | 1 | 100 | 100 | HPC Degradation |
| **FD002** | 6 | 1 | 260 | 259 | HPC Degradation (multi-condition) |
| **FD003** | 1 | 2 | 100 | 100 | HPC + Fan Degradation |
| **FD004** | 6 | 2 | 248 | 249 | HPC + Fan Degradation (multi-condition) |

---

## 🧱 Project Structure

```
predictive-maintenance-cmapss/
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── src/
│   └── utils/
│       └── cmapss.py
├── notebooks/
│   ├── 01_data_overview.ipynb
│   └── 02_preprocessing.ipynb
├── dashboard/
├── requirements.txt
└── README.md
```

---

## 🧠 Workflow Overview

| Step | Notebook | Description |
|------|-----------|-------------|
| **1** | `01_data_overview.ipynb` | Load raw data, visualize sensor trends, inspect correlations, identify useful sensors. |
| **2** | `02_preprocessing.ipynb` | Add RUL labels, drop bad sensors, scale features, save clean CSVs. |
| **3** | `03_modeling.ipynb` *(future)* | Build predictive models (LSTM, XGBoost). |
| **4** | `04_dashboard.ipynb` *(future)* | Visualize degradation & predictions. |

---

## 🧩 Step-1 Recap

You’ve already:
- Implemented robust data loading utilities  
- Loaded subset **FD001**  
- Explored:
  - Sensor behavior  
  - Engine variability  
  - Correlations with cycle & RUL  
- Identified sensor keep/drop lists  

---

## 🔜 Step-2 Preview

Next steps:
1. Add RUL labels  
2. Clean noisy/flat sensors  
3. Scale features  
4. Save processed datasets  

---

## ⚙️ Setup Instructions

```bash
git clone https://github.com/<yourname>/predictive-maintenance-cmapss.git
cd predictive-maintenance-cmapss
conda create -n cmapss python=3.11 -y
conda activate cmapss
pip install -r requirements.txt
jupyter notebook
```

---

## 📚 Reference

A. Saxena et al.,  
*“Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation,”* PHM08, Denver CO, 2008.

---

## 🏁 Why This Project Matters

Predictive maintenance bridges **AI and physical systems**, giving experience in:
- Multivariate time-series modeling  
- Feature engineering  
- End-to-end ML workflow  

Perfect for portfolios & interviews in ML, IoT, or MLOps.
