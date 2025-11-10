# streamlit_app_plain_features.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import json
from io import BytesIO
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Train plain Logistic (.pkl) - Fixed features", layout="wide")
st.title("Train plain LogisticRegression (.pkl) — fixed feature set (File_id dropped)")

st.markdown("""
This app requires the following numeric feature columns **(in your CSV)**:
`Band, Peak_amp, Energy, Rms, Kurotosis, F_peak, F_centroid, Bandwidth`
and a target column named `Label` (default mapping: `Healthy -> 0`, others -> 1).

**Important:** `File_id` will be automatically dropped if present.
The exported `.pkl` is a plain sklearn estimator (no dict/pipeline) so your EXE can load it and call `.predict(...)`.
""")

# expected features & label
REQUIRED_FEATURES = ["Band", "Peak_amp", "Energy", "Rms", "Kurotosis", "F_peak", "F_centroid", "Bandwidth"]
DEFAULT_LABEL_COL = "Label"
DEFAULT_HEALTHY_VALUE = "Healthy"

# file upload / local load
uploaded = st.file_uploader("Upload CSV with headers (or leave empty to load 'ae_features_all.csv' from app folder)", type=["csv"])
if uploaded is None:
    try:
        df = pd.read_csv("ae_features_all.csv")
        st.info("Loaded local file 'ae_features_all.csv'.")
    except FileNotFoundError:
        st.warning("No file uploaded and local 'ae_features_all.csv' not found. Please upload your CSV.")
        st.stop()
else:
    try:
        df = pd.read_csv(uploaded)
        st.success("Uploaded CSV loaded.")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()

st.subheader("CSV preview (first 6 rows)")
st.dataframe(df.head(6))

# Sidebar options
st.sidebar.header("Options")
label_col = st.sidebar.text_input("Label column name", value=DEFAULT_LABEL_COL)
healthy_value = st.sidebar.text_input("Value representing Healthy", value=DEFAULT_HEALTHY_VALUE)
test_size = st.sidebar.slider("Test set fraction", min_value=0.05, max_value=0.5, value=0.30, step=0.05)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)
use_scaler = st.sidebar.checkbox("Apply StandardScaler BEFORE training (Only enable if EXE expects scaled input)", value=False)
if use_scaler:
    st.sidebar.warning("If you enable scaling, your EXE MUST apply the SAME scaler before calling predict.")

# Validate presence of required columns; drop File_id automatically if present
if 'File_id' in df.columns:
    st.write("Dropping 'File_id' column (non-feature).")
    df = df.drop(columns=['File_id'])

missing = [c for c in REQUIRED_FEATURES + [label_col] if c not in df.columns]
if missing:
    st.error(f"Missing required columns in uploaded CSV: {missing}")
    st.stop()

# Build X (exact order) and y
X = df[REQUIRED_FEATURES].copy()
y_raw = df[label_col].copy()

# Map label to 0/1
def map_label(v):
    try:
        if isinstance(v, str):
            return 0 if v.strip() == healthy_value else 1
        else:
            return 0 if str(v) == str(healthy_value) else 1
    except Exception:
        return 1

y = y_raw.apply(map_label).astype(int)

st.write("Feature columns used (in order):", REQUIRED_FEATURES)
st.write("Label counts (after mapping):")
st.write(pd.Series(y).value_counts().to_dict(), " (0 = Healthy, 1 = Damaged)")

# Train button
if st.button("Train & Export plain Logistic .pkl"):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=float(test_size), random_state=int(random_state), stratify=y.values
        )
    except Exception as e:
        st.error(f"train_test_split failed: {e}")
        st.stop()

    scaler = None
    if use_scaler:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Train plain logistic (no pipeline)
    try:
        model = LogisticRegression(random_state=int(random_state), max_iter=2000)
        model.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test)[:, 1]
    except Exception:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    st.success(f"Training finished — Accuracy: {acc:.4f}")

    st.write("### Classification report")
    st.text(classification_report(y_test, y_pred))

    st.write("### Confusion matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4,3))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, int(val), ha='center', va='center', color='black')
    plt.colorbar(im, ax=ax)
    st.pyplot(fig)

    # ROC AUC if possible
    if y_prob is not None and len(np.unique(y_test)) == 2:
        fpr, tpr, thr = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        st.write(f"ROC AUC: {roc_auc:.4f}")
        fig2, ax2 = plt.subplots(figsize=(5,4))
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax2.plot([0,1],[0,1], linestyle="--", color="gray")
        ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.legend()
        st.pyplot(fig2)

    # Save plain model (only the estimator) for EXE
    suggested_name = "logistic_regression_model_plain.pkl"
    pkl_name = st.text_input("Filename for plain model (will be plain estimator .pkl)", value=suggested_name)
    if not pkl_name.endswith(".pkl"):
        pkl_name = pkl_name + ".pkl"

    buf = BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pkl_name}">Download plain estimator (.pkl) for EXE</a>'
    st.markdown(href, unsafe_allow_html=True)
    st.success("Plain estimator saved — this file contains only the sklearn estimator (callable .predict).")

    # Save metadata JSON with feature order & scaler flag
    meta = {"feature_columns": REQUIRED_FEATURES, "used_scaler": bool(use_scaler)}
    meta_bytes = json.dumps(meta).encode("utf-8")
    b64m = base64.b64encode(meta_bytes).decode()
    meta_name = pkl_name.replace(".pkl", "_meta.json")
    hrefm = f'<a href="data:application/octet-stream;base64,{b64m}" download="{meta_name}">Download metadata (.json)</a>'
    st.markdown(hrefm, unsafe_allow_html=True)

    st.info("⚠️ IMPORTANT: Ensure your EXE sends features to the model in **exactly this column order**. If you enable scaling, EXE must apply same scaler before calling predict (not recommended).")
