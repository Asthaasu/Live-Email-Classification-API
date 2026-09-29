"""
Phase 6: Explainability Dashboard
Run with: streamlit run src/app_dashboard.py
"""

import os
import pickle

import numpy as np
import pandas as pd
import shap
import streamlit as st

from gmail_features import extract_email_features

# -----------------------------------------------------------------
# Paths — model, scaler, and feature order all come from Phase 5
# -----------------------------------------------------------------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_balanced.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODEL_DIR, "phase5_feature_names.pkl")


@st.cache_resource
def load_artifacts():
    missing = [p for p in [MODEL_PATH, SCALER_PATH, FEATURE_NAMES_PATH] if not os.path.exists(p)]
    if missing:
        st.error(
            "Missing required file(s): "
            + ", ".join(missing)
            + ". Run `python src/train_models.py` first — it saves the model, "
            "scaler, and feature order together."
        )
        st.stop()

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURE_NAMES_PATH, "rb") as f:
        feature_names = pickle.load(f)

    return model, scaler, feature_names


model, scaler, feature_names = load_artifacts()

# -----------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------
st.set_page_config(page_title="Spam Detector v2.0", layout="wide")
st.title("🚨 Spam Detector v2.0 — Live Explainability Dashboard")
st.caption(f"Model: `xgb_balanced.pkl` | Scaler: `scaler.pkl` | Features: {len(feature_names)}")

debug_mode = st.sidebar.checkbox("🔧 Debug mode (show raw + scaled features)")

col1, col2 = st.columns(2)
with col1:
    subject = st.text_input("Email Subject", value="Urgent: Verify your account NOW")
with col2:
    sender = st.text_input("From (sender)", value="noreply+test@suspicious-bank.com")

body = st.text_area("Email Body (optional)", height=120)

if st.button("Analyze Email", type="primary"):
    # --- Extract features as a dict ---
    raw_features = extract_email_features(subject, sender, body)

    # --- Force exact training column order (critical fix) ---
    # feature_names came straight from phase5_feature_names.pkl, i.e.
    # the exact column order the scaler + model were fit on.
    missing_keys = [f for f in feature_names if f not in raw_features]
    if missing_keys:
        st.error(
            f"extract_email_features() is missing keys the model expects: {missing_keys}. "
            "gmail_features.py may have changed since training — retrain to sync."
        )
        st.stop()

    X_raw = np.array([[raw_features[f] for f in feature_names]])

    # --- Apply the SAME scaler used at training time ---
    X_scaled = scaler.transform(X_raw)

    if debug_mode:
        st.sidebar.subheader("Raw feature vector (training order)")
        st.sidebar.dataframe(
            pd.DataFrame(X_raw, columns=feature_names).T.rename(columns={0: "raw"})
        )
        st.sidebar.subheader("Scaled feature vector (fed to model)")
        st.sidebar.dataframe(
            pd.DataFrame(X_scaled, columns=feature_names).T.rename(columns={0: "scaled"})
        )

    # --- Predict on SCALED features ---
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]
    spam_confidence = proba[1] * 100
    legit_confidence = proba[0] * 100

    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Classification", "🚨 SPAM" if pred == 1 else "✅ LEGITIMATE")
    m2.metric("Spam Confidence", f"{spam_confidence:.1f}%")
    m3.metric("Legitimate Confidence", f"{legit_confidence:.1f}%")

    # --- SHAP explanation — also on SCALED features, matching training ---
    st.subheader("Why this prediction?")
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)

        if isinstance(shap_values, list):
            values_for_class = shap_values[int(pred)][0]
        else:
            values_for_class = shap_values[0]

        contributions = (
            pd.DataFrame({"Feature": feature_names, "Impact": values_for_class})
            .sort_values("Impact", key=abs, ascending=False)
            .head(10)
        )
        st.bar_chart(contributions.set_index("Feature"))
    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

st.markdown("---")
st.write("**v2.0 Improvements over v1.0:**")
st.write(
    """
- Live Gmail integration (real emails, not a static Kaggle dataset)
- Feature engineering beyond TF-IDF
- MLflow-tracked model comparison (RF vs XGBoost)
- SHAP explainability — see exactly why each email is flagged
- Deployed REST API + this dashboard
"""
)
