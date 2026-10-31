# app.py — Churn Risk Advisor (MPhil / Research Edition)
# Ultimate Fix: Dynamic feature matching for XGBoost to prevent mismatch errors.

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
import plotly.graph_objects as go

# --- Optional libraries ---
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
    from sklearn.inspection import permutation_importance
    HAS_SK = True
except Exception:
    HAS_SK = False

# -----------------------------------------------------------------------------
# 0. Page & constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Churn Risk Advisor — Research Edition",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "2.2-mphil-ultimate"
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
# 1. Provenance & artifact loading (Paths fixed to root)
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

@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts(model_path: str = "churn_model.joblib", meta_path: str = "model_meta.json"):
    model = joblib.load(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    provenance = {
        "app_version": APP_VERSION,
        "git_commit": _git_commit(),
        "model_sha256_16": _sha256(model_path),
        "meta_sha256_16": _sha256(meta_path),
        "loaded_at_utc": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
    }
    return model, meta, provenance

model, meta, PROVENANCE = load_artifacts()
COLS = meta.get("feature_columns", [])
MODEL_NAME = meta.get("model_name", type(model).__name__)
CV_AUC = float(meta.get("cv_auc", float("nan")))
DEFAULT_THRESHOLD = float(meta.get("threshold", 0.30))
UPLIFT = float(meta.get("uplift", 0.15))


# -----------------------------------------------------------------------------
# 2. Dynamic Scoring (Fixes XGBoost Feature Mismatch)
# -----------------------------------------------------------------------------
def prepare_input(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[RAW_COLS].copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    
    # 1. Get the exact features the model expects
    expected_features = None
    if hasattr(model, "get_booster"):
        expected_features = model.get_booster().feature_names
    elif hasattr(model, "feature_names_in_"):
        expected_features = model.feature_names_in_.tolist()
    elif hasattr(model, "steps"):
        try:
            expected_features = model[:-1].get_feature_names_out().tolist()
        except Exception:
            pass
            
    if not expected_features:
        expected_features = COLS

    # 2. Check if the model expects one-hot encoded features (e.g., 'gender_Male')
    if expected_features and any("_" in f for f in expected_features):
        # Encode the raw columns dynamically
        cat_cols_to_encode = [c for c in CATEGORICAL_COLS if c in df.columns]
        df_encoded = pd.get_dummies(df, columns=cat_cols_to_encode, dtype=float)
        
        # Reindex to perfectly match model's expected features
        df_final = df_encoded.reindex(columns=expected_features, fill_value=0)
        return df_final
    
    # 3. If the model accepts native categoricals (recent XGBoost versions)
    else:
        for col in CATEGORICAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype('category')
        if expected_features:
            df = df[expected_features]
        return df

def score(raw: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare_input(raw))[:, 1]

def score_with_uncertainty(raw: pd.DataFrame, n_boot: int = 200, seed: int = 42) -> dict:
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
        "sd": float(draws.std()),
        "n_boot": n_boot,
    }

# -----------------------------------------------------------------------------
# 3. Decision theory & counterfactuals
# -----------------------------------------------------------------------------
def risk_band(p: float, thr: float) -> str:
    if p >= thr: return "HIGH"
    if p >= thr / 2: return "WATCH"
    return "LOW"

def expected_cost(p: float, cost_contact: float, cost_churn: float) -> dict:
    cost_contact_ = cost_contact + p * (1 - UPLIFT) * cost_churn
    cost_not_ = p * cost_churn
    return {
        "cost_if_contact": cost_contact_,
        "cost_if_not": cost_not_,
        "delta": cost_contact_ - cost_not_,
        "optimal_action": "Contact" if cost_contact_ < cost_not_ else "No action",
    }

def contrastive_counterfactuals(customer: pd.DataFrame, p_base: float, top_k: int = 6) -> pd.DataFrame:
    candidates = [
        ("Contract", "One year"), ("Contract", "Two year"),
        ("PaymentMethod", "Credit card (automatic)"), ("PaperlessBilling", "No"),
    ]
    if customer.at[0, "InternetService"] != "No":
        for s in SERVICES:
            candidates.append((s, "Yes"))
        candidates.append(("InternetService", "DSL"))

    rows = []
    for f, v in candidates:
        if customer.at[0, f] == v: continue
        alt = customer.copy()
        alt[f] = v
        q = float(score(alt)[0])
        rows.append({
            "Feature": f, "Value": v, "New p": round(q, 4),
            "Δp": round(q - p_base, 4),
            "Relative Δ": f"{(q - p_base) / max(p_base, 1e-9):+.1%}",
        })
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Δp").head(top_k)

# -----------------------------------------------------------------------------
# 4. Fairness
# -----------------------------------------------------------------------------
def fairness_report(df: pd.DataFrame, thr: float, y_col: str | None = None):
    rows = []
    for attr in PROTECTED_ATTRS:
        if attr not in df.columns: continue
        for val, g in df.groupby(attr, dropna=False):
            r = {
                "attribute": attr, "group": str(val), "n": len(g),
                "flag_rate": float((g["p_churn"] >= thr).mean()),
                "mean_p": float(g["p_churn"].mean()),
            }
            if y_col and y_col in g.columns and g[y_col].nunique() == 2:
                try:
                    r["auc"] = float(roc_auc_score(g[y_col], g["p_churn"]))
                    r["brier"] = float(brier_score_loss(g[y_col], g["p_churn"]))
                except Exception:
                    pass
            rows.append(r)
    table = pd.DataFrame(rows)
    if table.empty: return table, pd.DataFrame()
    
    di = (
        table.groupby("attribute")["flag_rate"]
        .agg(lambda s: s.min() / max(s.max(), 1e-9))
        .rename("disparate_impact_ratio")
        .reset_index()
    )
    return table, di

# -----------------------------------------------------------------------------
# 5. Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### ⚙️ Configuration  \n`app {APP_VERSION}` · `commit {PROVENANCE['git_commit']}`")
    with st.expander("Model provenance", expanded=False):
        st.json(PROVENANCE)
        st.caption(f"**{MODEL_NAME}** · CV AUC ≈ {CV_AUC:.3f}")

    st.header("Customer profile")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    services = []
    if internet != "No":
        services = st.multiselect("Add-on services", SERVICES, default=["OnlineSecurity"])
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
    n_boot = st.select_slider("Bootstrap draws", options=[50, 100, 200, 500, 1000], value=200)


# -----------------------------------------------------------------------------
# 6. Build customer row
# -----------------------------------------------------------------------------
row = {
    "gender": gender, "SeniorCitizen": int(senior),
    "Partner": "Yes" if partner else "No",
    "Dependents": "Yes" if dependents else "No",
    "tenure": tenure, "PhoneService": phone, "MultipleLines": "No",
    "InternetService": internet, "Contract": contract,
    "PaperlessBilling": paperless, "PaymentMethod": payment,
    "MonthlyCharges": monthly, "TotalCharges": tenure * monthly,
}
for s in SERVICES:
    row[s] = ("No internet service" if internet == "No" else ("Yes" if s in services else "No"))
customer = pd.DataFrame([row])

# -----------------------------------------------------------------------------
# 7. Header & Tabs
# -----------------------------------------------------------------------------
st.title("🛰️ Churn Risk Advisor — Research Edition")
st.caption(f"{MODEL_NAME} · CV AUC ≈ {CV_AUC:.3f} · threshold = {threshold:.2f} · decision-support only")

tab_single, tab_batch, tab_fair, tab_explain, tab_methods = st.tabs(
    ["👤 One customer", "📦 Batch scoring", "⚖️ Fairness audit", "🔬 Explainability", "📖 Methods & model card"]
)

# =============================================================================
# TAB 1 — Single customer
# =============================================================================
with tab_single:
    unc = score_with_uncertainty(customer, n_boot=n_boot)
    p, lo, hi = unc["p"], unc["ci_low"], unc["ci_high"]
    band = risk_band(p, threshold)
    ec = expected_cost(p, cost_contact, cost_churn)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Churn probability", f"{p:.1%}", f"95% band [{lo:.1%}, {hi:.1%}]")
    c2.metric("Risk band", band)
    c3.metric("Expected-net Δ (contact − not)", f"${ec['delta']:+,.0f}", help="Negative = contacting saves money.")
    c4.metric("Recommended action", ec["optimal_action"])
    st.progress(min(max(p, 0.0), 1.0))

    st.divider()
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.subheader("Counterfactual: what would change the risk?")
        cf = contrastive_counterfactuals(customer, p)
        if len(cf):
            st.dataframe(cf, hide_index=True, use_container_width=True)
            st.caption("Model-implied **associations**, not causal effects.")
        else:
            st.info("No single-feature counterfactual changes the profile materially.")

    with col_b:
        st.subheader("Decision-theoretic threshold")
        grid = np.linspace(0.01, 0.99, 99)
        rows = []
        for t in grid:
            act = p >= t
            cost = (cost_contact + p * (1 - UPLIFT) * cost_churn) if act else p * cost_churn
            rows.append({"threshold": t, "expected_cost": cost, "action": "Contact" if act else "No action"})
        dd = pd.DataFrame(rows)
        best = dd.loc[dd["expected_cost"].idxmin()]
        fig = px.line(dd, x="threshold", y="expected_cost", labels={"expected_cost": "Expected cost ($)"})
        fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"current={threshold:.2f}")
        fig.add_vline(x=float(best["threshold"]), line_dash="dot", line_color="green", annotation_text=f"optimal={best['threshold']:.2f}")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# TAB 2 — Batch scoring
# =============================================================================
with tab_batch:
    st.write("Upload a CSV with the original Telco columns.")
    file = st.file_uploader("Customer file (CSV)", type="csv")
    data = None
    if file:
        data = pd.read_csv(file)
        missing = [c for c in RAW_COLS if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)
            data["p_churn"] = score(data)
            data["risk_band"] = data["p_churn"].map(lambda x: risk_band(x, threshold))
            data["contact"] = np.where(data["p_churn"] >= threshold, "Yes", "")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rows", f"{len(data):,}")
            m2.metric("Flagged (≥ threshold)", f"{(data['contact'] == 'Yes').sum():,}", f"{(data['contact'] == 'Yes').mean():.1%}")
            m3.metric("Mean p", f"{data['p_churn'].mean():.1%}")
            m4.metric("High-band share", f"{(data['risk_band'] == 'HIGH').mean():.1%}")

            st.dataframe(data.sort_values("p_churn", ascending=False).head(200), use_container_width=True, hide_index=True)
            st.download_button("⬇️ Download scored CSV", data.to_csv(index=False).encode("utf-8"), "scored.csv", "text/csv")

            if "Churn" in data.columns and HAS_SK:
                y = data["Churn"].map({"Yes": 1, "No": 0})
                if y.notna().all():
                    st.subheader("Out-of-sample evaluation on this file")
                    e1, e2, e3 = st.columns(3)
                    e1.metric("AUC", f"{roc_auc_score(y, data['p_churn']):.3f}")
                    e2.metric("Brier", f"{brier_score_loss(y, data['p_churn']):.3f}")
                    e3.metric("Avg precision", f"{average_precision_score(y, data['p_churn']):.3f}")

# =============================================================================
# TAB 3 — Fairness audit
# =============================================================================
with tab_fair:
    st.write("Subgroup diagnostics on the uploaded batch (or the single profile).")
    audit_df = data.copy() if isinstance(data, pd.DataFrame) and "p_churn" in data.columns else customer.assign(p_churn=[score(customer)[0]])
    y_col = "Churn" if "Churn" in audit_df.columns else None
    if y_col:
        audit_df["__y"] = audit_df[y_col].map({"Yes": 1, "No": 0})
        y_col = "__y"

    table, di = fairness_report(audit_df, thr=threshold, y_col=y_col)
    if not table.empty:
        st.subheader("Subgroup metrics")
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.subheader("Disparate impact ratio")
        st.dataframe(di, use_container_width=True, hide_index=True)
    else:
        st.info("Not enough data for fairness audit. Upload a batch CSV.")

# =============================================================================
# TAB 4 — Explainability
# =============================================================================
with tab_explain:
    st.write("Local and global explanations.")
    X_cust = prepare_input(customer)
    
    st.subheader("Global Feature Importance (XGBoost)")
    if hasattr(model, "feature_importances_"):
        if hasattr(model, "steps"):
            clf = model[-1]
            if hasattr(clf, "feature_importances_"):
                importances = clf.feature_importances_
                try: feat_names = model[:-1].get_feature_names_out()
                except: feat_names = [f"f{i}" for i in range(len(importances))]
            else: importances, feat_names = [], []
        else:
            importances = model.feature_importances_
            if hasattr(model, "get_booster"):
                feat_names = model.get_booster().feature_names
                if not feat_names: feat_names = [f"f{i}" for i in range(len(importances))]
            else: feat_names = [f"f{i}" for i in range(len(importances))]
        
        if len(importances) > 0:
            imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
            imp_df = imp_df.sort_values("Importance", ascending=False).head(15)
            fig = px.bar(imp_df.sort_values("Importance"), x="Importance", y="Feature", orientation="h")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Model does not expose feature importances directly.")

    st.divider()
    st.subheader("SHAP (local)")
    if HAS_SHAP:
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer(X_cust)
            vals = sv.values[0]
            if len(vals.shape) > 1: vals = vals[:, 1] 
            
            shap_df = pd.DataFrame({"Feature": X_cust.columns, "SHAP Value": vals})
            shap_df = shap_df.reindex(shap_df["SHAP Value"].abs().sort_values(ascending=False).index).head(12)
            fig = px.bar(shap_df.sort_values("SHAP Value"), x="SHAP Value", y="Feature", orientation="h")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP unavailable: {e}")
    else:
        st.info("Install `shap` to enable SHAP explanations.")

# =============================================================================
# TAB 5 — Methods & model card
# =============================================================================
with tab_methods:
    st.header("Model card & methods appendix")
    st.markdown(f"""
**Model name:** {MODEL_NAME}  
**App version:** `{APP_VERSION}`  
**Model sha256 (16):** `{PROVENANCE['model_sha256_16']}`  
**Loaded at:** `{PROVENANCE['loaded_at_utc']}`

### Intended use
Decision support for retention outreach. Not an autonomous decision-maker.

### Limitations
1. Associations ≠ effects — counterfactual deltas need causal validation.
2. Uplift parameter (`{UPLIFT}`) is an assumption.
""")

    st.divider()
    st.subheader("Export audit log (thesis appendix)")
    audit_payload = {
        "provenance": PROVENANCE,
        "current_inputs": row,
        "current_p": float(score(customer)[0]),
        "threshold": threshold,
    }
    st.download_button(
        "⬇️ Download audit JSON",
        json.dumps(audit_payload, indent=2).encode("utf-8"),
        "audit_log.json", "application/json",
    )
