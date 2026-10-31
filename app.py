# app.py — Churn Risk Advisor (MPhil Research Edition - FAST & OPTIMIZED)

import json
import hashlib
import subprocess
import datetime as dt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

# --- Optional libraries ---
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
    HAS_SK = True
except Exception:
    HAS_SK = False

# -----------------------------------------------------------------------------
# 0. Page & constants
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Churn Risk Advisor", layout="wide")

APP_VERSION = "2.3-fast"
RAW_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']
PROTECTED_ATTRS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']
CATEGORICAL_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                    'PaperlessBilling', 'PaymentMethod']

# -----------------------------------------------------------------------------
# 1. Provenance & artifact loading
# -----------------------------------------------------------------------------
def _sha256(path: str) -> str:
    p = Path(path)
    if not p.exists(): return "MISSING"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()[:16]

def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "n/a"

@st.cache_resource(show_spinner="Loading model...")
def load_artifacts(model_path: str = "churn_model.joblib", meta_path: str = "model_meta.json"):
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    provenance = {
        "app_version": APP_VERSION, "git_commit": _git_commit(),
        "model_sha256": _sha256(model_path), "loaded_at": dt.datetime.utcnow().isoformat() + "Z"
    }
    return model, meta, provenance

model, meta, PROVENANCE = load_artifacts()
COLS = meta.get("feature_columns", [])
MODEL_NAME = meta.get("model_name", type(model).__name__)
CV_AUC = float(meta.get("cv_auc", float("nan")))
DEFAULT_THRESHOLD = float(meta.get("threshold", 0.30))
UPLIFT = float(meta.get("uplift", 0.15))

# -----------------------------------------------------------------------------
# 2. Scoring
# -----------------------------------------------------------------------------
def prepare_input(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[RAW_COLS].copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    
    expected_features = None
    if hasattr(model, "get_booster"): expected_features = model.get_booster().feature_names
    elif hasattr(model, "feature_names_in_"): expected_features = model.feature_names_in_.tolist()
    elif hasattr(model, "steps"):
        try: expected_features = model[:-1].get_feature_names_out().tolist()
        except: pass
            
    if not expected_features: expected_features = COLS

    if expected_features and any("_" in f for f in expected_features):
        cat_cols_to_encode = [c for c in CATEGORICAL_COLS if c in df.columns]
        df_encoded = pd.get_dummies(df, columns=cat_cols_to_encode, dtype=float)
        return df_encoded.reindex(columns=expected_features, fill_value=0)
    else:
        for col in CATEGORICAL_COLS:
            if col in df.columns: df[col] = df[col].astype('category')
        if expected_features: df = df[expected_features]
        return df

def score(raw: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare_input(raw))[:, 1]

# ---------------------------------------------------------
# FAST UNCERTAINTY (Optimized for Streamlit Cloud)
# ---------------------------------------------------------
def score_with_uncertainty(raw: pd.DataFrame, n_boot: int = 30, seed: int = 42) -> dict:
    # n_boot reduced to 30 for instant response
    rng = np.random.default_rng(seed)
    base = float(score(raw)[0])
    draws = []
    for _ in range(n_boot):
        noise = rng.normal(0.0, 0.02, size=1)
        perturbed = raw.copy()
        for c in ["tenure", "MonthlyCharges", "TotalCharges"]:
            perturbed[c] = np.clip(perturbed[c] * (1 + noise[0]), 0, None)
        draws.append(float(score(perturbed)[0]))
    draws = np.array(draws)
    return {
        "p": base,
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }

def risk_band(p: float, thr: float) -> str:
    if p >= thr: return "HIGH"
    if p >= thr / 2: return "WATCH"
    return "LOW"

def expected_cost(p: float, cost_contact: float, cost_churn: float) -> dict:
    cost_contact_ = cost_contact + p * (1 - UPLIFT) * cost_churn
    cost_not_ = p * cost_churn
    return {
        "delta": cost_contact_ - cost_not_,
        "optimal_action": "Contact" if cost_contact_ < cost_not_ else "No action",
    }

def contrastive_counterfactuals(customer: pd.DataFrame, p_base: float, top_k: int = 5) -> pd.DataFrame:
    candidates = [("Contract", "One year"), ("Contract", "Two year"), ("PaymentMethod", "Credit card (automatic)")]
    if customer.at[0, "InternetService"] != "No":
        candidates.append(("TechSupport", "Yes"))
    
    rows = []
    for f, v in candidates:
        if customer.at[0, f] == v: continue
        alt = customer.copy()
        alt[f] = v
        q = float(score(alt)[0])
        rows.append({"Feature": f, "Value": v, "New p": round(q, 3), "Δp": round(q - p_base, 3)})
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Δp").head(top_k)

# -----------------------------------------------------------------------------
# 3. Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Customer profile")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    services = st.multiselect("Add-on services", SERVICES, default=["OnlineSecurity"]) if internet != "No" else []
    payment = st.selectbox("Payment", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
    paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)

    with st.expander("Demographics"):
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        senior = st.checkbox("Senior citizen")
        partner = st.checkbox("Has partner")
        dependents = st.checkbox("Has dependents")

    st.divider()
    st.subheader("Decision parameters")
    threshold = st.slider("Contact threshold", 0.05, 0.95, DEFAULT_THRESHOLD, 0.01)
    cost_contact = st.number_input("Cost of retention contact ($)", 0.0, 500.0, 25.0, step=5.0)
    cost_churn = st.number_input("LTV lost per churn ($)", 0.0, 5000.0, 800.0, step=50.0)

# -----------------------------------------------------------------------------
# 4. Build customer row
# -----------------------------------------------------------------------------
row = {
    "gender": gender, "SeniorCitizen": int(senior), "Partner": "Yes" if partner else "No",
    "Dependents": "Yes" if dependents else "No", "tenure": tenure, "PhoneService": phone, 
    "MultipleLines": "No", "InternetService": internet, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": tenure * monthly,
}
for s in SERVICES:
    row[s] = ("No internet service" if internet == "No" else ("Yes" if s in services else "No"))
customer = pd.DataFrame([row])

# -----------------------------------------------------------------------------
# 5. Header & Tabs
# -----------------------------------------------------------------------------
st.title("🛰️ Churn Risk Advisor")
st.caption(f"{MODEL_NAME} · CV AUC ≈ {CV_AUC:.3f} · threshold = {threshold:.2f}")

tab_single, tab_batch, tab_explain = st.tabs(["👤 One customer", "📦 Batch scoring", "🔬 Explainability"])

# =============================================================================
# TAB 1 — Single customer (FAST LOAD)
# =============================================================================
with tab_single:
    unc = score_with_uncertainty(customer) # Fast execution
    p, lo, hi = unc["p"], unc["ci_low"], unc["ci_high"]
    band = risk_band(p, threshold)
    ec = expected_cost(p, cost_contact, cost_churn)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn probability", f"{p:.1%}", f"Range [{lo:.1%}, {hi:.1%}]")
    c2.metric("Risk band", band)
    c3.metric("Expected-net Δ", f"${ec['delta']:+,.0f}")
    c4.metric("Action", ec["optimal_action"])
    
    st.progress(min(max(p, 0.0), 1.0))
    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("What changes the risk?")
        cf = contrastive_counterfactuals(customer, p)
        if len(cf): 
            st.dataframe(cf, hide_index=True, use_container_width=True)
        else: 
            st.info("No major single-feature changes detected.")

    with col_b:
        st.subheader("Decision-theoretic threshold")
        grid = np.linspace(0.01, 0.99, 50)
        costs = [(cost_contact + p * (1 - UPLIFT) * cost_churn) if p >= t else p * cost_churn for t in grid]
        best_t = grid[np.argmin(costs)]
        
        fig = px.line(x=grid, y=costs, labels={"x": "Threshold", "y": "Expected Cost ($)"})
        fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Current")
        fig.add_vline(x=float(best_t), line_dash="dot", line_color="green", annotation_text="Optimal")
        fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2 — Batch scoring
# =============================================================================
with tab_batch:
    file = st.file_uploader("Upload CSV", type="csv")
    if file:
        data = pd.read_csv(file)
        if not all(c in data.columns for c in RAW_COLS):
            st.error("Missing required columns in CSV.")
        else:
            data["p_churn"] = score(data)
            data["risk_band"] = data["p_churn"].map(lambda x: risk_band(x, threshold))
            st.metric("Flagged for contact", f"{(data['p_churn'] >= threshold).sum():,} / {len(data):,}")
            st.dataframe(data.sort_values("p_churn", ascending=False).head(100), use_container_width=True)
            st.download_button("Download CSV", data.to_csv(index=False).encode("utf-8"), "scored.csv", "text/csv")

# =============================================================================
# TAB 3 — Explainability (LAZY LOADING - Only runs when clicked)
# =============================================================================
with tab_explain:
    st.write("Feature importance and local explanations.")
    X_cust = prepare_input(customer)
    
    # Global Importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_names = model.get_booster().feature_names if hasattr(model, "get_booster") else [f"f{i}" for i in range(len(importances))]
        imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances}).sort_values("Importance", ascending=False).head(15)
        st.plotly_chart(px.bar(imp_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h", height=400), use_container_width=True)

    # SHAP (Only runs when tab is open)
    st.divider()
    st.subheader("SHAP (Local Contribution)")
    if HAS_SHAP:
        with st.spinner("Calculating SHAP values..."):
            try:
                explainer = shap.TreeExplainer(model)
                sv = explainer(X_cust)
                vals = sv.values[0]
                if len(vals.shape) > 1: vals = vals[:, 1] 
                
                shap_df = pd.DataFrame({"Feature": X_cust.columns, "SHAP Value": vals})
                shap_df = shap_df.reindex(shap_df["SHAP Value"].abs().sort_values(ascending=False).index).head(10)
                st.plotly_chart(px.bar(shap_df.sort_values("SHAP Value"), x="SHAP Value", y="Feature", orientation="h"), use_container_width=True)
            except Exception as e:
                st.info(f"SHAP error: {e}")
    else:
        st.info("Install `shap` in requirements.txt.")
