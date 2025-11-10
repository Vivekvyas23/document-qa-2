# streamlit_app_save_plain_model.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from io import BytesIO
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="AE → Plain Logistic .pkl (EXE-compatible)", layout="wide")
st.title("Build & Export plain LogisticRegression `.pkl` (EXE-compatible)")

st.markdown("""
This app trains a **plain** `sklearn.linear_model.LogisticRegression` and saves the estimator object directly
(with `joblib.dump(model, ...)`) — **no dict, no pipeline, no wrapper** — so it can be loaded by your EXE and `.predict()` can be called.

**Important:** By default this app does **not** apply scaling or preprocessing before training, to match the behavior of your original script (so the produced `.pkl` is compatible with your EXE).
""")

# ----------------------------
# Upload or load local CSV
# ----------------------------
uploaded = st.file_uploader("Upload features CSV (ae_features_all.csv) or leave empty to load local file named 'ae_features_all.csv'", type=["csv"])
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        st.success("Uploaded CSV loaded.")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    try:
        data = pd.read_csv("ae_features_all.csv")
        st.info("Loaded local 'ae_features_all.csv'.")
    except FileNotFoundError:
        st.warning("No file uploaded and 'ae_features_all.csv' not found in app folder. Please upload your CSV.")
        st.stop()

st.subheader("Preview data (first 8 rows)")
st.dataframe(data.head(8))

# ----------------------------
# Options
# ----------------------------
st.sidebar.header("Options (make sure these match what EXE expects)")
drop_fileid = st.sidebar.checkbox("Drop column 'File_id' if present (recommended)", value=True)
label_col = st.sidebar.text_input("Label column name", value="Label")
label_value_healthy = st.sidebar.text_input("Value that represents 'Healthy'", value="Healthy")
test_size = st.sidebar.slider("Test set fraction", 0.05, 0.5, value=0.30, step=0.05)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

# IMPORTANT: do not scale by default (EXE expects plain model)
st.sidebar.info("This app trains WITHOUT scaling by default so the saved .pkl is EXE-compatible.")
use_scaler = st.sidebar.checkbox("Apply StandardScaler (WARNING: likely breaks EXE compatibility)", value=False)
if use_scaler:
    st.sidebar.warning("If you enable scaling, your EXE must also apply the SAME scaling before predict. Only enable if EXE expects scaled input.")

# ----------------------------
# Prepare X and y (mimic user's script)
# ----------------------------
df = data.copy()

# Drop File_id if present and user requested
if drop_fileid and 'File_id' in df.columns:
    df = df.drop(columns=['File_id'])

# Validate label column
if label_col not in df.columns:
    st.error(f"Label column '{label_col}' not found in the CSV. Please set the correct label column name in sidebar.")
    st.stop()

# Create target y: Healthy -> 0, else -> 1
def map_label(v):
    try:
        if isinstance(v, str):
            return 0 if v.strip() == label_value_healthy else 1
        else:
            return 0 if str(v) == str(label_value_healthy) else 1
    except Exception:
        return 1

y = df[label_col].apply(map_label).astype(int)
# Build X by dropping the label
X = df.drop(columns=[label_col], errors='ignore')

# Drop non-numeric feature cols
non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
if len(non_numeric) > 0:
    st.warning(f"Dropping non-numeric feature columns: {non_numeric}")
    X = X.drop(columns=non_numeric)

if X.shape[1] == 0:
    st.error("No numeric feature columns available for training. Check your CSV.")
    st.stop()

st.write("Using feature columns:", list(X.columns))
st.write(f"Label distribution: {pd.Series(y).value_counts().to_dict()} (0=Healthy, 1=Damaged)")

# ----------------------------
# Train button
# ----------------------------
if st.button("Train & Export plain Logistic .pkl (EXE-compatible)"):
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=float(test_size), random_state=int(random_state), stratify=y.values
        )
    except Exception as e:
        st.error(f"train_test_split failed: {e}")
        st.stop()

    # Optional scaling (not recommended if EXE expects plain model)
    scaler = None
    if use_scaler:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Train plain LogisticRegression (no pipeline)
    try:
        model = LogisticRegression(random_state=int(random_state), max_iter=1000)
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

    if y_prob is not None:
        fpr, tpr, thr = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        st.write(f"ROC AUC: {roc_auc:.4f}")
        fig2, ax2 = plt.subplots(figsize=(5,4))
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax2.plot([0,1], [0,1], linestyle='--', color='gray')
        ax2.set_xlabel("FPR"); ax2.set_ylabel("TPR"); ax2.legend()
        st.pyplot(fig2)

    # ----------------------------
    # Save plain model ONLY (EXE expects estimator with .predict)
    # ----------------------------
    pkl_filename = st.text_input("Filename to save model as (will be plain estimator .pkl)", value="logistic_regression_model.pkl")
    if not pkl_filename.endswith(".pkl"):
        pkl_filename = pkl_filename + ".pkl"

    # Save model to bytes buffer and provide download link
    buffer = BytesIO()
    # IMPORTANT: save ONLY the model object (not scaler, not dict) so EXE can load and predict
    joblib.dump(model, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()

    st.write("✅ Plain estimator saved. Download below (this .pkl can be loaded with joblib.load(...) and .predict(...) called directly).")
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pkl_filename}">Download plain model .pkl</a>'
    st.markdown(href, unsafe_allow_html=True)

    # Also optionally save a note file containing feature columns & whether scaling was used
    info = {
        "feature_columns": list(X.columns),
        "used_scaler": bool(use_scaler)
    }
    info_buf = BytesIO()
    joblib.dump(info, info_buf)
    info_buf.seek(0)
    b64i = base64.b64encode(info_buf.read()).decode()
    info_name = pkl_filename.replace(".pkl", "_meta.pkl")
    href2 = f'<a href="data:application/octet-stream;base64,{b64i}" download="{info_name}">Download metadata (.pkl) with feature column list</a>'
    st.markdown(href2, unsafe_allow_html=True)

    st.info("⚠️ Make sure your EXE expects the same feature column order as printed above. If you enabled scaling, your EXE must apply identical scaling BEFORE calling predict (not recommended if you cannot change EXE).")
