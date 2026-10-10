# applied-ai-project1-churn - Customer Churn Prediction
**Name: Muhammad Musa**

This repository contains Project 1 work from Week 1 to Week 4 for Applied AI course.

## Week 1 - EDA (Exploratory Data Analysis)
- **Notebook:** `week1-eda.ipynb`
- **Dataset:** Telco Customer Churn Dataset - 7043 rows, 21 columns
- **Work Done:**
    - Missing values handling in TotalCharges column
    - Converted TotalCharges to numeric type
    - Univariate Analysis - Churn distribution, Tenure, MonthlyCharges
    - Bivariate Analysis - Contract type vs Churn, InternetService vs Churn, PaymentMethod vs Churn
- **Key Findings from my analysis:**
    - Total Churn Rate: 26.54%
    - Month-to-month contract churn rate is highest ~42%
    - Tenure less than 12 months has higher churn
    - Fiber optic and Electronic check payment customers churn more
    - Customers without OnlineSecurity and TechSupport churn more

## Week 2 - Machine Learning Models
# Week 2 — Telco Customer Churn Modeling

Applied AI · Project 1 · ML-Powered Customer Analytics

A complete machine learning pipeline comparing **Logistic Regression**,
**Decision Trees**, and **Random Forests** on the Telco Customer Churn
dataset (7,043 customers, 26.5% churn rate).

---

## 🎯 What This Project Does

Builds, evaluates, and interprets three classification models on real
customer churn data. Focuses on **why accuracy is misleading** under
class imbalance, and how **threshold choice is a business decision**,
not a technical default.

---

## 📊 Dataset

- **Source:** [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Rows:** 7,043 customers
- **Features:** 20 (demographics, services, contract, billing)
- **Target:** `Churn` (binary)
- **Class balance:** 26.5% churn / 73.5% stay

---

## 🧹 Preprocessing Pipeline

1. **Clean** — `TotalCharges` (11 blanks at `tenure=0`) → numeric → median imputation
2. **Encode** — one-hot encoding with `drop_first=True` (avoid dummy trap)
3. **Split** — 80/20 stratified holdout on `Churn`
4. **Scale** — `StandardScaler` fitted **only on training set** (prevents leakage)

All wrapped in a `sklearn.pipeline.Pipeline`.

---

## 🤖 Models Trained

| Model | Core Idea |
|---|---|
| **DummyClassifier** (baseline) | Always predicts majority class |
| **Logistic Regression** | Sigmoid + cross-entropy, linear in log-odds |
| **Logistic Regression (balanced)** | `class_weight='balanced'` to reweight minority class |
| **Decision Tree (depth 5)** | Greedy impurity splits (Gini) |
| **Random Forest** | 300 trees, `max_features='sqrt'`, OOB validation |

---

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Baseline (always STAY) | 0.735 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | **0.807** | 0.658 | 0.567 | 0.609 | **0.842** |
| LogReg + balanced | 0.740 | 0.506 | **0.783** | 0.615 | 0.841 |
| Decision Tree (depth 5) | 0.796 | 0.632 | 0.551 | 0.589 | 0.829 |
| Random Forest | **0.807** | 0.673 | 0.529 | 0.593 | **0.842** |

**Highlight:** The baseline achieves 73.5% accuracy while catching
**zero churners** — proving accuracy alone is worthless here.

---

## 🔍 Key Findings

### 1. Accuracy is misleading under class imbalance
A model that never predicts churn scores 73.5% accuracy. For churn,
**recall matters more** — a missed churner is lost revenue.

### 2. Top churn drivers (from odds ratios)
- **Risk factors:** Month-to-month contract, Fiber optic internet, high `MonthlyCharges`
- **Protective factors:** Two-year contract (odds ×0.23), longer `tenure` (odds ×0.95/month)

### 3. Class imbalance handling
`class_weight='balanced'` raised **recall 57% → 78%** at the cost of
**precision 66% → 51%**. For churn campaigns, this is the right trade-off.

### 4. Threshold is a business decision
With FN 6× more costly than FP (PKR 6,000 vs 1,000), the cost-optimal
threshold is **t = 0.14**. For a balanced campaign, `t = 0.30` works well.

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.3 | 0.50 | 0.78 | 0.61 |
| 0.5 (default) | 0.64 | 0.57 | 0.60 |
| 0.7 | 0.74 | 0.33 | 0.46 |

### 5. Trees: depth 5 is the sweet spot
Train accuracy climbs to 99% at depth 15, but test accuracy drops to 72%
— classic **overfitting**. Depth 4–5 is optimal here.

### 6. Random Forest generalizes
OOB score ≈ test score → forest isn't overfitting. AUC matches LogReg
(0.842), but Random Forest additionally provides **feature importances**.

---

## 📂 Files in This Repo

| File | Description |
|---|---|
| `week2_churn_models.ipynb` | Full Kaggle notebook (all code, plots, insights) |
| `model_compare.csv` | Side-by-side metrics for all 5 models |
| `odds_ratios.csv` | Logistic Regression coefficients + odds ratios |
| `feature_importance.csv` | Random Forest permutation importance |
| `insights.md` | Written summary of findings |

---

## 🛠️ Tech Stack

- **Python 3.7**
- **pandas, numpy** — data manipulation
- **scikit-learn** — pipelines, models, metrics
- **matplotlib, seaborn** — visualization
- **Jupyter / Kaggle Notebooks** — environment
---

## ▶️ How to Run

### On Kaggle (recommended)
1. Open the notebook: **[YOUR-KAGGLE-LINK]**
2. Click **Copy & Edit**
3. **Add Input** → search "Telco Customer Churn" → add dataset
4. **Run All**

### Locally
```bash
git clone https://github.com/khan123kh/applied-ai-week2.git
cd applied-ai-week2
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook week2_churn_models.ipynb
- To be added in Week 2
- Notebook will be: `week2-ml-models.ipynb`

## Week 3 - Optimization
- To be added in Week 3
- Notebook will be: `week3-optimization.ipynb`
## week 3 and 4 works
# Applied AI Project 1 - Churn Risk Advisor

**Live Demo:** https://applied-ai-project1-churn-h4taotbpneuw5xy6e57fw.streamlit.app
**Repository:** https://github.com/musa100179/applied-ai-project1-churn

### Week 3 & Week 4 - Combined Work

#### Week 3: Model Development
- **Dataset:** Telco Customer Churn dataset
- **Preprocessing:** Handled missing values, encoded categorical features (Contract, PaymentMethod, InternetService etc.), scaled numerical features.
- **Model Used:** XGBClassifier
- **Evaluation:** 
    - CV AUC: 0.845
    - Train/Test Split: 80/20
    - Metrics: Precision, Recall, F1-Score
- **Artifacts Saved:**
    - `churn_model.joblib` - Final trained model
    - `model_meta.json` - Feature list and threshold

#### Week 4: Deployment (Streamlit App)
- **App Name:** Churn Risk Advisor v1.0
- **Features Implemented:**
    1.  **One Customer Scoring:** Left panel se Tenure, Contract, Monthly Charges change karke live risk dekho.
    2.  **Batch Scoring:** `sample_customers.csv` upload karke hazaron customers ka risk ek saath.
    3.  **What would change the risk?:** Agar contract ko Two year kiya jaye ya Payment method change ho to risk kitna kam hoga, ye suggest karta hai.
    4.  **Risk Bands:** LOW / MEDIUM / HIGH with Action (Contact now / Monitor).

#### How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
## Week 4 - Final Report and Deployment
- To be added in Week 4

## Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Scikit-learn

##  APP Link
https://applied-ai-project1-churn-h4taotbpneuwv5xy6e576w.streamlit.app/

## Author
Muhammad Musa - Musa100179
