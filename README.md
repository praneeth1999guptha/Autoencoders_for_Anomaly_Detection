# Autoencoders_for_Anomaly_Detection
Implement autoencoder and explore its application for a real-world problem related to anomaly detection.
# Autoencoders for Anomaly Detection (AWS CloudWatch, NAB)

_Unsupervised multi-metric time‑series anomaly detection using dense and LSTM autoencoders in PyTorch._

This repository accompanies the notebook **`notebooks/Autoencoders for Anomaly Detection.ipynb`**, which trains autoencoders to detect anomalies in AWS CloudWatch–style metrics derived from the **Numenta Anomaly Benchmark (NAB)** dataset.

> **Highlight:** Final LSTM Autoencoder reached a **pseudo accuracy ≈ 96.8%** on the test set when flagging anomalies using the **85th‑percentile reconstruction‑error threshold**, with stable train/validation loss curves and clearly separated error distributions (see notebook figures).

---

## ✨ What this project does

- Aggregates several CloudWatch‑like CSV time series into a single aligned dataframe (forward‑filled), then engineers features.
- Trains two unsupervised models:
  - **Small Dense Autoencoder** (tabular features)
  - **LSTM Autoencoder** (sequence modeling with a sliding window)
- Evaluates via **reconstruction error (MSE / RMSE)** and flags anomalies by thresholding error (percentile/IQR).
- Includes exploratory visuals: **ACF plots**, **recent-window heatmaps**, loss curves, and error histograms.

---

## 📦 Data

- **Source:** *Numenta Anomaly Benchmark (NAB)* — Kaggle: https://www.kaggle.com/datasets/boltzmannbrain/nab  
- **What you need:** A directory of CSV files whose filenames contain these patterns (the notebook uses `glob` to find them):
  - `*cpu_utilization*.csv` → **CPU**
  - `*disk_write_bytes*.csv` → **Disk**
  - `*network_in*.csv` → **Network**
  - `*elb_request_count*.csv` → **ELB**
  - `*rds_cpu_utilization*.csv` → **RDS**
  - `*grok_asg_anomaly*.csv` → **ASG**
- The notebook merges them into a multi-metric table and saves:
  - `cloudwatch_multimetric_aggregated.csv` (aligned, forward‑filled, sorted by timestamp)

> Set the path at the top of the notebook: `DATA_FOLDER = "/path/to/your/data"`.

---

## 🧪 Feature engineering & preprocessing

Implemented in `preprocess_features_v2`:
- Marks **sparse metrics** (`Disk`, `Network`, `ELB`) with
  - a **binary indicator** (`*_bin`) for non‑zero activity, and
  - a **log‑magnitude** feature (`*_log = log1p(value)` clipped to high percentiles).
- Keeps **dense metrics** (`CPU`, `RDS`, `ASG`) continuous.
- Scales non‑binary columns with **RobustScaler** (resistant to outliers).

Optional data augmentation: **jitter** a subset of training samples with small Gaussian noise (σ≈0.03).

---

## 🧠 Models

### 1) Small Dense Autoencoder
- **Encoder/Decoder:** Linear → ReLU → Dropout; bottleneck size ≈ **48**
- **Loss:** MSE; **Optimizer:** Adam
- Example training settings: `epochs=30`, `batch_size=64`, `lr≈9.76e‑4`, `weight_decay≈1.86e‑6`, `dropout≈0.22`

### 2) LSTM Autoencoder
- **Sequence window:** `seq_length=10`
- **Encoder/Decoder:** LSTM (hidden_dim=32, num_layers=1) → bottleneck **16**
- **Loss:** MSE; **Optimizer:** Adam; **Scheduler:** ReduceLROnPlateau
- Example training settings: `epochs=20`, `batch_size=64`, `lr≈1.15e‑4`, `weight_decay≈5.97e‑5`, `dropout≈0.10`

> Many hyperparameters in the notebook were selected via **Optuna** trials.

---

## 🧮 Evaluation & anomaly scoring

- Compute **per‑sample reconstruction error** (MSE).  
- Choose a threshold:
  - **Percentile method:** e.g., **85th percentile** of train errors (default shown)
  - **IQR method:** `Q3 + 1.5×IQR` from train errors
- Flag anomalies where `error > threshold`.
- Visuals: loss curves, **train vs test error histograms** with threshold line.
- Reported in the notebook:
  - **Pseudo accuracy ≈ 96.79%** (e.g., `5269/5444` test windows ≤ 85th‑percentile threshold).  
    _“Pseudo” means we don’t use ground‑truth labels here; we evaluate how well the chosen threshold separates typical from atypical behavior._

---

## 🚀 Getting started

### 1) Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt  # see below
```

### 2) Data
- Download NAB and place the relevant CSVs in a folder (see patterns above).
- Open the notebook and set `DATA_FOLDER` to that folder path.

### 3) Run
```bash
jupyter notebook notebooks/Autoencoders\ for\ Anomaly\ Detection.ipynb
```
Run cells in order:
1. **Step 1: Data preparation** (merge CSVs → `cloudwatch_multimetric_aggregated.csv`)
2. **Step 2: Autoencoder model building** (choose dense or LSTM)
3. **Step 3: Evaluation and analysis** (thresholding & visuals)

> GPU is auto‑detected; CPU also works (slower).

---



---

## 🔖 Citation & acknowledgments

- Lavin, A., & Ahmad, S. (2015). **Evaluating Real‑Time Anomaly Detection Algorithms – the Numenta Anomaly Benchmark (NAB).**
- **Dataset:** *Numenta Anomaly Benchmark (NAB)* — Kaggle: https://www.kaggle.com/datasets/boltzmannbrain/nab
- Thanks to the PyTorch, pandas, scikit‑learn, statsmodels, and seaborn communities.

---
