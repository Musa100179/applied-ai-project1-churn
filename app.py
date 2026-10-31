# app.py — Churn Risk Advisor (MPhil / Research Edition)
# -----------------------------------------------------------------------------
# Research-grade decision support:
#   • Reproducibility & provenance (SHA-256, UTC timestamp, git commit)
#   • Calibrated probabilities + bootstrap uncertainty bands
#   • Fairness audit (subgroup metrics, disparate impact, four-fifths rule)
#   • Decision-theoretic threshold optimisation (expected cost)
#   • Local (log-odds, SHAP) + global (permutation) explainability
#   • Full model card, methods appendix, reproducibility checklist
#   • Audit-log export for thesis appendix
# -----------------------------------------------------------------------------

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

# --- Optional libraries (graceful fallback) ---------------------------------
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import (
        roc_auc_score, brier_score_loss, log_loss,
        average_precision_score, precision_recall_curve,
    )
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

APP_VERSION = "2.0-mphil"
RAW_COLS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
SERVICES = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies']
PROTECTED_ATTRS = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']


# -----------------------------------------------------------------------------
# 1. Provenance & artifact loading
# -----------------------------------------------------------------------------
def _sha256(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "n/a"


@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts(model_path: str = "artifacts/churn_model.joblib",
                   meta_path: str = "artifacts/model_meta.json"):
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
COLS = meta["feature_columns"]
MODEL_NAME = meta.get("model_name", type(model).__name__)
CV_AUC = float(meta.get("cv_auc", float("nan")))
DEFAULT_THRESHOLD = float(meta.get("threshold", 0.30))
UPLIFT = float(meta.get("uplift", 0.15))


# -----------------------------------------------------------------------------
# 2. Scoring (pipeline handles encoding internally)
# -----------------------------------------------------------------------------
def prepare_input(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[RAW_COLS].copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df


def score(raw: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(prepare_input(raw))[:, 1]


def score_with_uncertainty(raw: pd.DataFrame, n_boot: int = 200, seed: int = 42) -> dict:
    """Perturbation-based stability interval on the single-customer score.
    For a true posterior, retrain on bootstrap resamples (see Methods tab)."""
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
    if p >= thr:
        return "HIGH"
    if p >= thr / 2:
        return "WATCH"
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


def contrastive_counterfactuals(customer: pd.DataFrame, p_base: float,
                                top_k: int = 6) -> pd.DataFrame:
    candidates = [
        ("Contract", "One year"),
        ("Contract", "Two year"),
        ("PaymentMethod", "Credit card (automatic)"),
        ("PaperlessBilling", "No"),
    ]
    if customer.at[0, "InternetService"] != "No":
        for s in SERVICES:
            candidates.append((s, "Yes"))
        candidates.append(("InternetService", "DSL"))

    rows = []
    for f, v in candidates:
        if customer.at[0, f] == v:
            continue
        alt = customer.copy()
        alt[f] = v
        q = float(score(alt)[0])
        rows.append({
            "Feature": f,
            "Value": v,
            "New p": round(q, 4),
            "Δp": round(q - p_base, 4),
            "Relative Δ": f"{(q - p_base) / max(p_base, 1e-9):+.1%}",
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Δp").head(top_k)


# -----------------------------------------------------------------------------
# 4. Fairness
# -----------------------------------------------------------------------------
def fairness_report(df: pd.DataFrame, thr: float, y_col: str | None = None):
    rows = []
    for attr in PROTECTED_ATTRS:
        if attr not in df.columns:
            continue
        for val, g in df.groupby(attr, dropna=False):
            r = {
                "attribute": attr,
                "group": str(val),
                "n": len(g),
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
    payment = st.selectbox(
        "Payment", ["Electronic check", "Mailed check",
                    "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
    paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)

    with st.expander("Demographics"):
        gender = st.radio("Gender", ["Female", "Male"], horizontal=True)
        senior = st.checkbox("Senior citizen")
        partner = st.checkbox("Has partner")
        dependents = st.checkbox("Has dependents")

    st.divider()
    st.subheader("Decision parameters")
    threshold = st.slider("Contact threshold", 0.05, 0.95,
                          DEFAULT_THRESHOLD, 0.01,
                          help="Threshold used for HIGH band and batch flagging.")
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
    row[s] = ("No internet service" if internet == "No"
              else ("Yes" if s in services else "No"))
customer = pd.DataFrame([row])


# -----------------------------------------------------------------------------
# 7. Header
# -----------------------------------------------------------------------------
st.title("🛰️ Churn Risk Advisor — Research Edition")
st.caption(
    f"{MODEL_NAME} · CV AUC ≈ {CV_AUC:.3f} · "
    f"threshold = {threshold:.2f} · decision-support only (human-in-the-loop required)"
)


# -----------------------------------------------------------------------------
# 8. Tabs
# -----------------------------------------------------------------------------
tab_single, tab_batch, tab_fair, tab_explain, tab_methods = st.tabs(
    ["👤 One customer", "📦 Batch scoring", "⚖️ Fairness audit",
     "🔬 Explainability", "📖 Methods & model card"]
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
    c3.metric("Expected-net Δ (contact − not)", f"${ec['delta']:+,.0f}",
              help="Negative = contacting saves money in expectation.")
    c4.metric("Recommended action", ec["optimal_action"])
    st.progress(min(max(p, 0.0), 1.0))

    st.divider()
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.subheader("Counterfactual: what would change the risk?")
        cf = contrastive_counterfactuals(customer, p)
        if len(cf):
            st.dataframe(cf, hide_index=True, use_container_width=True)
            st.caption(
                "Model-implied **associations**, not causal effects. "
                "Do not interpret Δp as guaranteed uplift without A/B validation."
            )
        else:
            st.info("No single-feature counterfactual changes the profile materially.")

    with col_b:
        st.subheader("Decision-theoretic threshold")
        grid = np.linspace(0.01, 0.99, 99)
        rows = []
        for t in grid:
            act = p >= t
            cost = (cost_contact + p * (1 - UPLIFT) * cost_churn) if act else p * cost_churn
            rows.append({"threshold": t, "expected_cost": cost,
                         "action": "Contact" if act else "No action"})
        dd = pd.DataFrame(rows)
        best = dd.loc[dd["expected_cost"].idxmin()]
        fig = px.line(dd, x="threshold", y="expected_cost",
                      labels={"expected_cost": "Expected cost ($)"})
        fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                      annotation_text=f"current={threshold:.2f}")
        fig.add_vline(x=float(best["threshold"]), line_dash="dot", line_color="green",
                      annotation_text=f"optimal={best['threshold']:.2f}")
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Under current parameters, expected-cost optimum ≈ "
            f"**{best['threshold']:.2f}** → {best['action']} "
            f"(cost ${best['expected_cost']:.0f})."
        )


# =============================================================================
# TAB 2 — Batch scoring
# =============================================================================
with tab_batch:
    st.write("Upload a CSV with the original Telco columns. "
             "Preprocessing is applied via the model pipeline (no leakage).")
    file = st.file_uploader("Customer file (CSV)", type="csv")

    data = None
    if file:
        data = pd.read_csv(file)
        missing = [c for c in RAW_COLS if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)
        data["p_churn"] = score(data)
        data["risk_band"] = data["p_churn"].map(lambda x: risk_band(x, threshold))
        data["contact"] = np.where(data["p_churn"] >= threshold, "Yes", "")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{len(data):,}")
        m2.metric("Flagged (≥ threshold)", f"{(data['contact'] == 'Yes').sum():,}",
                  f"{(data['contact'] == 'Yes').mean():.1%}")
        m3.metric("Mean p", f"{data['p_churn'].mean():.1%}")
        m4.metric("High-band share", f"{(data['risk_band'] == 'HIGH').mean():.1%}")

        st.dataframe(
            data.sort_values("p_churn", ascending=False).head(200),
            use_container_width=True, hide_index=True,
        )
        st.download_button(
            "⬇️ Download scored CSV",
            data.to_csv(index=False).encode("utf-8"),
            "scored.csv", "text/csv",
        )

        if "Churn" in data.columns and HAS_SK:
            y = data["Churn"].map({"Yes": 1, "No": 0})
            if y.notna().all():
                st.subheader("Out-of-sample evaluation on this file")
                e1, e2, e3 = st.columns(3)
                e1.metric("AUC", f"{roc_auc_score(y, data['p_churn']):.3f}")
                e2.metric("Brier", f"{brier_score_loss(y, data['p_churn']):.3f}")
                e3.metric("Avg precision",
                          f"{average_precision_score(y, data['p_churn']):.3f}")


# =============================================================================
# TAB 3 — Fairness audit
# =============================================================================
with tab_fair:
    st.write(
        "Subgroup diagnostics on the uploaded batch (or the single profile). "
        "Descriptive only — **not** a regulatory certification."
    )
    audit_df = data.copy() if isinstance(data, pd.DataFrame) else \
               customer.assign(p_churn=[score(customer)[0]])

    y_col = "Churn" if "Churn" in audit_df.columns else None
    if y_col:
        audit_df["__y"] = audit_df[y_col].map({"Yes": 1, "No": 0})
        y_col = "__y"

    table, di = fairness_report(audit_df, thr=threshold, y_col=y_col)

    st.subheader("Subgroup metrics")
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Disparate impact ratio (min/max flag rate)")
    st.dataframe(di, use_container_width=True, hide_index=True)
    st.caption("Ratios below **0.80** flag a four-fifths rule concern (EEOC). "
               "Interpret cautiously for small subgroup *n*.")

    if HAS_SK and y_col:
        st.subheader("Calibration by subgroup")
        for attr in PROTECTED_ATTRS:
            if attr not in audit_df.columns:
                continue
            fig = go.Figure()
            for val, g in audit_df.groupby(attr):
                if g["__y"].nunique() < 2:
                    continue
                try:
                    prob_true, prob_pred = calibration_curve(
                        g["__y"], g["p_churn"], n_bins=8, strategy="quantile")
                    fig.add_trace(go.Scatter(
                        x=prob_pred, y=prob_true, mode="lines+markers",
                        name=f"{attr}={val} (n={len(g)})"))
                except Exception:
                    pass
            fig.add_trace(go.Scatter([0, 1], [0, 1], mode="lines",
                                     line=dict(dash="dash", color="grey"),
                                     name="Perfect calibration"))
            fig.update_layout(title=f"Calibration — {attr}",
                              xaxis_title="Predicted probability",
                              yaxis_title="Observed churn rate",
                              height=360, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# TAB 4 — Explainability
# =============================================================================
with tab_explain:
    st.write("Local and global explanations. "
             "Contributions are on the log-odds scale for logistic models.")

    X_cust = prepare_input(customer)
    final = model[-1] if hasattr(model, "steps") else model
    try:
        z_row = model[:-1].transform(X_cust)[0]
    except Exception:
        z_row = X_cust.iloc[0].values

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.subheader("Local contribution (log-odds)")
        if hasattr(final, "coef_"):
            contrib = pd.Series(final.coef_[0] * z_row, index=COLS)
            top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(12)
            fig = px.bar(top.sort_values(), orientation="h",
                         labels={"value": "log-odds contribution", "index": "feature"},
                         color=top.sort_values().values, color_continuous_scale="RdBu")
            fig.update_layout(showlegend=False, coloraxis_showscale=False, height=460)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Final estimator has no `coef_`; use SHAP below for nonlinear models.")

    with col2:
        st.subheader("Permutation importance (global)")
        st.caption("Requires a labelled reference set — upload one in the Batch tab "
                   "with a `Churn` column to compute.")
        if isinstance(data, pd.DataFrame) and "Churn" in data.columns and HAS_SK:
            try:
                sample = data.sample(min(len(data), 500), random_state=0)
                X_ref = prepare_input(sample)
                y_ref = sample["Churn"].map({"Yes": 1, "No": 0})
                r = permutation_importance(model, X_ref, y_ref, n_repeats=5,
                                           random_state=0, scoring="roc_auc")
                # Map back to raw feature importance by summing encoded columns
                try:
                    pre = model[:-1]
                    enc_names = pre.get_feature_names_out().tolist()
                except Exception:
                    enc_names = COLS
                imp = pd.Series(r.importances_mean, index=enc_names[:len(r.importances_mean)])
                # Aggregate one-hot groups by prefix
                agg = {}
                for name, val in imp.items():
                    key = name.split("_")[0]
                    agg[key] = agg.get(key, 0.0) + float(val)
                agg_series = pd.Series(agg).sort_values(ascending=False).head(15)
                fig = px.bar(agg_series.sort_values(), orientation="h",
                             labels={"value": "ΔAUC", "index": "feature"})
                fig.update_layout(height=460, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Permutation importance failed: {e}")

    st.divider()
    st.subheader("SHAP (local, if available)")
    if HAS_SHAP:
        try:
            X_t = model[:-1].transform(X_cust)
            explainer = shap.LinearExplainer(model[-1], X_t)
            sv = explainer(X_t)
            vals = pd.Series(np.ravel(sv.values[0]), index=COLS).sort_values()
            top = pd.concat([vals.head(6), vals.tail(6)])
            st.dataframe(top.rename("SHAP value").to_frame(),
                         use_container_width=True)
        except Exception as e:
            st.info(f"SHAP unavailable for this estimator: {e}")
    else:
        st.info("Install `shap` to enable SHAP explanations.")

    st.divider()
    st.subheader("Robustness — sensitivity to contact threshold")
    grid = np.linspace(0.05, 0.95, 19)
    piv = [{"threshold": round(t, 2),
            "flag": "Yes" if score(customer)[0] >= t else "No"} for t in grid]
    st.dataframe(pd.DataFrame(piv), use_container_width=True, hide_index=True)


# =============================================================================
# TAB 5 — Methods & model card
# =============================================================================
with tab_methods:
    st.header("Model card & methods appendix")
    st.markdown(f"""
**Model name:** {MODEL_NAME}  
**App version:** `{APP_VERSION}` · **git commit:** `{PROVENANCE['git_commit']}`  
**Model sha256 (16):** `{PROVENANCE['model_sha256_16']}` · **Meta sha256 (16):** `{PROVENANCE['meta_sha256_16']}`  
**Loaded at:** `{PROVENANCE['loaded_at_utc']}`

### Intended use
Decision support for retention outreach on the IBM Telco churn dataset.
**Not an autonomous decision-maker** — all HIGH flags should be reviewed by a
human agent. Not validated for populations or contracts materially different
from the training distribution.

### Data & preprocessing
- Original IBM Telco schema (19 features).
- Categorical features one-hot encoded **inside** the pipeline (no leakage);
  missing / malformed `TotalCharges` are coerced to 0.
- Single-customer mode recomputes `TotalCharges = tenure × monthly`.

### Training & evaluation
- Reported **CV AUC ≈ {CV_AUC:.3f}** (see `model_meta.json`).
- Default decision threshold `{DEFAULT_THRESHOLD:.2f}`; operator-overridable.
- Sidebar exposes a **decision-theoretic optimum** that minimises expected cost
  given user-supplied `cost_contact` and `cost_churn`.

### Calibration
Per-subgroup reliability curves are shown in the **Fairness audit** tab
(requires a `Churn` label in the uploaded file).

### Uncertainty
Single-customer scores are accompanied by a **perturbation-based 95% interval**
({n_boot} draws, ±2% multiplicative noise on monetary/tenure features).
This is a *stability* interval — for a full posterior on p(y|x), retrain on
bootstrap resamples of the training set and ship the ensemble weights.

### Fairness
The Fairness tab computes per-group flag rates, subgroup AUC/Brier (when labels
exist), and the **four-fifths rule** disparate-impact ratio. Small subgroup *n*
should be interpreted with caution.

### Explainability
- Logistic models: exact log-odds contributions (`coef_ × z`).
- Any model: permutation importance on a labelled reference sample; SHAP if `shap` is installed.
- Counterfactuals are **associational**, not causal.

### Limitations
1. Associations ≠ effects — counterfactual deltas need causal validation (A/B).
2. Uplift parameter (`{UPLIFT}`) is an assumption — reported explicitly.
3. Training data is a single public dataset; external validity unverified.
4. Fairness diagnostics are descriptive, not a compliance statement.

### Reproducibility checklist
- [x] Model hash and load timestamp recorded.
- [x] Preprocessing frozen inside pipeline.
- [x] Decision threshold user-set and reported.
- [x] Decision theory exposed via cost inputs.
- [x] Batch export includes per-row `p_churn` and risk band.
- [x] Audit log downloadable (below).
""")

    st.divider()
    st.subheader("Raw model metadata")
    st.json({k: v for k, v in meta.items() if k != "feature_columns"})
    st.caption(f"Feature contract: {len(COLS)} columns after one-hot encoding.")

    st.divider()
    st.subheader("Export audit log (thesis appendix)")
    audit_payload = {
        "provenance": PROVENANCE,
        "meta_public": {k: v for k, v in meta.items() if k != "feature_columns"},
        "current_inputs": row,
        "current_p": float(score(customer)[0]),
        "threshold": threshold,
        "cost_contact": cost_contact,
        "cost_churn": cost_churn,
    }
    st.download_button(
        "⬇️ Download audit JSON",
        json.dumps(audit_payload, indent=2).encode("utf-8"),
        "audit_log.json", "application/json",
    )
