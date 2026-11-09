# CleanDS: Task-Aware Automated Data Cleaning Pipeline

CleanDS is an intelligent, goal-oriented dataset preprocessing engine. Instead of applying one-size-fits-all heuristic scripts, CleanDS adapts its preprocessing operations based on the **intended downstream machine learning or analytical objective**.

---

## 🏗️ System Architecture & Workflow

CleanDS structures data cleaning as a cascading, task-aware machine learning pipeline:

```
                      [ User Natural Language Goal ]
                                    │
                                    ▼
                     ┌───────────────────────────────┐
                     │   1. Typo-Resilient Vector    │
                     │          RAG Matcher          │
                     │  (Char n-grams + Cosine Sim)  │
                     └──────────────┬────────────────┘
                                    │ Generates Cleaning Policy
                                    ▼
[ Raw CSV / Excel ] ──► ┌───────────────────────────────┐
                        │   2. 1D-CNN Column Role       │
                        │         Classifier            │
                        │    (Character-Level CNN)      │
                        └───────────┬───────────────────┘
                                    │ Classifies Column Types
                                    ▼
                        ┌───────────────────────────────┐
                        │   3. Outlier & Anomaly Engine │
                        │        (One-Class SVM)        │
                        └───────────┬───────────────────┘
                                    │ Identifies Multivariate Anomalies
                                    ▼
                        ┌───────────────────────────────┐
                        │   4. Intelligent Imputation   │
                        │    (Random Forest / MICE)     │
                        └───────────┬───────────────────┘
                                    │
                                    ▼
                        [ Cleaned Output Dataset ]
```

---

## 🧠 Core Machine Learning Components

### 1. Offline Vector RAG Matcher (Goal Interpretation)
* **What it does:** Translates natural language goals (e.g., *"Train XGBoost for equipment failure prediction"*) into an actionable cleaning policy.
* **Typo Resilience via Sub-Word Character n-grams:** Instead of exact whole-word matching, the retriever vectorizes text into word-boundary character n-grams ($N=3 \text{ to } 5$). Even if a query contains severe typos (e.g., `"predicitve maintanance on iot sensrs"`), the overlapping character sub-sequences ensure high cosine similarity to the correct playbook.
* **Why Vector Retrieval over an External LLM API:**
  * **Zero Latency:** Executes in under 2 milliseconds on CPU (vs. 2–4 seconds for an external LLM call).
  * **100% Offline & Free:** Requires zero API keys or active internet access.
  * **Zero Hallucination:** Directly binds user intent to rigorously defined, schema-safe cleaning configurations without inventing non-existent parameters.

### 2. 1D Convolutional Neural Network (Column Schema Discovery)
* **Architecture:** Character-level PyTorch `Conv1D` $\to$ `BatchNorm1D` $\to$ `ReLU` $\to$ `MaxPool1D` $\to$ `Conv1D` $\to$ `GlobalMaxPool` $\to$ `Dense` $\to$ `Dropout` $\to$ `Softmax`.
* **What it does:** Scans character-level punctuation and alphanumeric distributions in sample values to classify column semantics (`Date_Time`, `Currency`, `Identifier`, `Numeric`, `Categorical`).
* **Contribution:** Allows automated parsing (e.g., stripping currency symbols to floats, normalizing dates to ISO-8601, and detecting ID columns to prevent data leakage).

### 3. One-Class SVM with RBF Kernel (Multivariate Anomaly Detection)
* **What it does:** Unsupervised boundary estimation for detecting multidimensional outliers.
* **Mechanism:** Projects numeric features into a higher-dimensional reproducing kernel Hilbert space using the Radial Basis Function (RBF) kernel to enclose the "normal" data cluster.
* **Task Adaptation:** In standard ML mode, extreme outliers are clipped to prevent gradient divergence. In fraud/anomaly detection mode, records outside the hyperplane boundary are flagged with an `is_anomaly` feature without deleting valid minority patterns.

### 4. Random Forest / MICE (Intelligent Missing Data Imputation)
* **What it does:** Multivariate Imputation by Chained Equations (MICE).
* **Mechanism:** Rather than applying flat statistical averages (mean/median) which destroy natural feature variance, CleanDS models each feature with missing values as a function of all other features using an iterative ensemble of Random Forest regressors.

---

## 🎯 Supported Domain & General Playbooks

CleanDS comes pre-configured with 16 task-specific cleaning policies:

* **Machine Learning Readiness:** Clips extreme tails to 99th percentiles, drops memorizable identifiers, and applies MICE.
* **Anomaly & Fraud Detection:** Preserves every anomalous row, adds binary anomaly flags, and maintains transaction IDs.
* **Clinical Data Integrity:** Retains rare physiological readings while modeling vitals via multivariate regression.
* **Predictive Maintenance:** Preserves critical sensor spikes, interpolates short telemetry dropouts, and flags flatline readings.
* **Financial Accuracy, Customer Churn, Sales Forecasting, Supply Chain Optimization, and more.**

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/bhartalayush/CleanDS.git
cd CleanDS
pip install -r requirements.txt
```

### 2. Run the Interface
CleanDS includes a lightweight interactive UI for testing and inspection:
```bash
streamlit run app.py
```

---

## 📁 Repository Structure

```text
CleanDS/
├── app.py                      # Clean interface for dataset upload and processing
├── requirements.txt            # Minimal dependencies (pandas, scikit-learn, torch)
├── README.md                   # System documentation and architecture breakdown
│
├── core/
│   ├── cleaner.py              # Central pipeline engine orchestrating ML steps
│   ├── playbooks.py            # Knowledge base of 16 domain cleaning policies
│   └── profiler.py             # Dataset metric extraction (nulls, duplicates, types)
│
├── models/
│   ├── cnn_classifier.py       # 1D-CNN in PyTorch for character sequence analysis
│   ├── outlier_detector.py     # One-Class SVM with RBF kernel (+ IsolationForest fallback)
│   ├── rag_matcher.py          # Typo-resilient character n-gram vector retriever
│   └── weights/                # Serialized model weights (cnn_column_classifier.pth)
│
└── data/
    └── sample_dirty_data.csv   # Dirty benchmark dataset for evaluation
```
