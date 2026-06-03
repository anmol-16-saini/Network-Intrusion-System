# 🚀 NIDS Project — GitHub & Docker Deployment Guide

---

## PART 1 — Push to GitHub

### Step 1 — Install Git

Download and install Git from:
```
https://git-scm.com/download/win
```
During install, keep all default options. After install, verify:
```bash
git --version
```

---

### Step 2 — Configure Git (one-time setup)

Open terminal and run:
```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

---

### Step 3 — Create a GitHub account & new repository

1. Go to https://github.com and sign up / log in
2. Click the **+** icon (top right) → **New repository**
3. Fill in:
   - Repository name: `nids-network-intrusion-detection`
   - Description: `Unsupervised ML-based Network Intrusion Detection System`
   - Visibility: **Public** (or Private)
   - ❌ Do NOT check "Add README" (we'll push our own)
4. Click **Create repository**
5. Copy the repository URL shown — looks like:
   ```
   https://github.com/yourusername/nids-network-intrusion-detection.git
   ```

---

### Step 4 — Create a .gitignore file

Inside `C:\Projects\nids\`, create a file named `.gitignore` with this content:

```
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Virtual environments
nids_env/
venv/
env/
.env

# Data files (too large for GitHub)
*.csv
*.parquet
*.pkl
*.h5

# Jupyter checkpoints
.ipynb_checkpoints/

# OS files
.DS_Store
Thumbs.db

# Streamlit cache
.streamlit/
```

> ⚠️ This prevents uploading the dataset CSV and virtual environment (which are huge).

---

### Step 5 — Create a README.md

Create `README.md` in `C:\Projects\nids\`:

```markdown
# 🛡️ Network Intrusion Detection System (NIDS)

Unsupervised ML-based NIDS using PCA, MiniBatch K-Means,
Isolation Forest, One-Class SVM, and a Hybrid Ensemble —
built with Streamlit.

## Features
- Interactive dashboard with 5 analysis tabs
- Upload any CICIDS-format CSV dataset
- Configurable model parameters via sidebar
- EDA, PCA visualization, anomaly detection, model comparison

## Dataset
Download from Kaggle:
https://www.kaggle.com/datasets/chethuhn/network-intrusion-dataset

## Run Locally
```bash
conda create -n nids_env python=3.11 -y
conda activate nids_env
conda install scikit-learn pandas numpy matplotlib seaborn -y
pip install streamlit
streamlit run app.py
```

## Tech Stack
- Python 3.11
- Streamlit
- Scikit-learn (PCA, KMeans, IsolationForest, OneClassSVM)
- Pandas, NumPy, Matplotlib, Seaborn
```

---

### Step 6 — Initialize and push to GitHub

Open terminal in `C:\Projects\nids\` and run these commands one by one:

```bash
# Initialize git repo
git init

# Add all files
git add .

# First commit
git commit -m "Initial commit: NIDS Streamlit app"

# Link to your GitHub repo (replace URL with yours)
git remote add origin https://github.com/yourusername/nids-network-intrusion-detection.git

# Push to GitHub
git branch -M main
git push -u origin main
```

GitHub will ask for your username and password.
> ⚠️ Use a **Personal Access Token** as password (not your GitHub password):
> GitHub → Settings → Developer Settings → Personal Access Tokens → Generate new token (classic) → check `repo` scope → copy the token and paste it as password.

---

### Step 7 — Verify

Go to `https://github.com/yourusername/nids-network-intrusion-detection` — you should see all your files live!

---
---

## PART 2 — Dockerize the App

### Step 1 — Install Docker Desktop

Download from:
```
https://www.docker.com/products/docker-desktop/
```
Install and restart your PC. Verify:
```bash
docker --version
```

---

### Step 2 — Create a Dockerfile

Create a file named `Dockerfile` (no extension) in `C:\Projects\nids\`:

```dockerfile
# Base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY app.py .

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run the app
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
```

---

### Step 3 — Update requirements.txt for Docker

Make sure your `requirements.txt` looks exactly like this:

```
streamlit>=1.35.0
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
seaborn>=0.13.0
```

---

### Step 4 — Create a .dockerignore file

Create `.dockerignore` in `C:\Projects\nids\`:

```
nids_env/
__pycache__/
*.csv
*.pkl
*.h5
.git/
.gitignore
*.ipynb
```

---

### Step 5 — Build the Docker image

Open terminal in `C:\Projects\nids\` and run:

```bash
docker build -t nids-app .
```

This will take 3–5 minutes the first time (downloading base image + installing packages).

---

### Step 6 — Run the Docker container

```bash
docker run -p 8501:8501 nids-app
```

Then open your browser at:
```
http://localhost:8501
```

Your app is now running inside Docker! ✅

---

### Step 7 — Useful Docker commands

```bash
# List running containers
docker ps

# Stop the container
docker stop <container_id>

# Remove the container
docker rm <container_id>

# Remove the image
docker rmi nids-app

# Run in background (detached mode)
docker run -d -p 8501:8501 nids-app

# View logs
docker logs <container_id>
```

---

### Step 8 — Push Docker image to Docker Hub (optional)

Share your image publicly so anyone can run it with one command.

```bash
# Login to Docker Hub
docker login

# Tag your image (replace yourusername)
docker tag nids-app yourusername/nids-app:latest

# Push to Docker Hub
docker push yourusername/nids-app:latest
```

Anyone can then run your app with just:
```bash
docker run -p 8501:8501 yourusername/nids-app:latest
```

---
---

## Final Project File Structure

```
C:\Projects\nids\
├── app.py                  ← Streamlit application
├── requirements.txt        ← Python dependencies
├── Dockerfile              ← Docker build instructions
├── .dockerignore           ← Files to exclude from Docker
├── .gitignore              ← Files to exclude from Git
├── README.md               ← Project description
└── HOW_TO_RUN.md           ← Local setup guide
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Push changes to GitHub | `git add . && git commit -m "msg" && git push` |
| Build Docker image | `docker build -t nids-app .` |
| Run Docker container | `docker run -p 8501:8501 nids-app` |
| Run locally (conda) | `conda activate nids_env && streamlit run app.py` |
