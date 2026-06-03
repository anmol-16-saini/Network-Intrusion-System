# 🛡️ NIDS — Streamlit App: How to Run Guide

## Overview
This Streamlit application provides an interactive dashboard for the **Network Intrusion Detection System (NIDS)** project. It runs the full ML pipeline — data cleaning, EDA, PCA, MiniBatch K-Means clustering, Isolation Forest, One-Class SVM, and a Hybrid Ensemble — all from your browser.

---

## 📦 Prerequisites

- **Python 3.9 or higher** (3.10 / 3.11 recommended)
- **pip** (comes with Python)

> ⚠️ If you're on Python 3.13 and see scikit-learn import errors (as seen in the notebook), use Python 3.11 instead.

---

## 🗂️ Project Folder Structure

```
nids_app/
├── app.py              ← Main Streamlit application
└── requirements.txt    ← Python dependencies
```

---

## 🚀 Step-by-Step Setup

### Step 1 — Create a virtual environment (recommended)

**Windows:**
```bash
python -m venv nids_env
nids_env\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv nids_env
source nids_env/bin/activate
```

---

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

This installs: `streamlit`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`.

---

### Step 3 — Run the app

```bash
streamlit run app.py
```

Your browser will automatically open at:
```
http://localhost:8501
```

---

### Step 4 — Upload your dataset

1. In the **sidebar on the left**, click **"Browse files"** under *Upload Dataset (CSV)*.
2. Upload the CICIDS network traffic CSV file, e.g.:
   ```
   Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv
   ```
   > The dataset must contain a `Label` column (values like `BENIGN`, `PortScan`, etc.).

3. You can **download this dataset** from Kaggle:
   - Search: `chethuhn/network-intrusion-dataset` on Kaggle
   - Or direct URL: https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

---

### Step 5 — Configure parameters

In the sidebar, adjust:

| Parameter | Description | Default |
|-----------|-------------|---------|
| KMeans Clusters | Number of traffic behavior groups | 5 |
| PCA Components | Dimensions to reduce to | 10 |
| IF Contamination | Expected anomaly fraction (Isolation Forest) | 0.05 |
| One-Class SVM ν | Nu parameter controlling boundary tightness | 0.05 |
| Sample Size | Rows to use (smaller = faster) | 25,000 |

---

### Step 6 — Run the analysis

Click the **▶ Run Analysis** button. Results are cached — changing parameters and re-running is fast.

---

## 📊 Dashboard Tabs

| Tab | Content |
|-----|---------|
| **EDA** | Attack distribution, correlation heatmap, feature distributions, raw data preview |
| **PCA & Clustering** | Explained variance, 2D & 3D cluster scatter plots, attack % per cluster |
| **Anomaly Detection** | Isolation Forest, One-Class SVM, Hybrid model & Ground Truth PCA plots |
| **Model Comparison** | Accuracy, Precision, Recall, F1-Score across all models with bar charts |
| **Cluster Analysis** | Cluster risk table, stacked composition chart, silhouette score interpretation |

---

## ⚡ Performance Tips

- Start with **Sample Size = 10,000** for a quick test run (~30 seconds).
- For full dataset (213K+ rows), use **Sample Size = All** — expect 3–8 minutes depending on hardware.
- One-Class SVM is the slowest model; reduce sample size or increase ν if it's too slow.

---

## 🐛 Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` again |
| scikit-learn import errors on Python 3.13 | Switch to Python 3.11: `python3.11 -m venv nids_env` |
| App doesn't open in browser | Manually visit `http://localhost:8501` |
| `Label column not found` error | Ensure your CSV has a `Label` column (case-sensitive) |
| Memory error on large files | Reduce Sample Size in the sidebar |
| Port 8501 in use | Run `streamlit run app.py --server.port 8502` |

---

## 🔄 Stopping the App

Press **Ctrl + C** in the terminal where the app is running.

---

## 📋 Quick Reference (copy-paste commands)

```bash
# Setup (one-time)
python -m venv nids_env
nids_env\Scripts\activate          # Windows
# source nids_env/bin/activate     # macOS/Linux
pip install -r requirements.txt

# Run
streamlit run app.py
```
