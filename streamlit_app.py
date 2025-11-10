# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, RocCurveDisplay
)
import matplotlib.pyplot as plt
import joblib
import base64

st.set_page_config(page_title="Damage Detector Trainer", layout="wide")

st.title("🔧 Damage-Detection Model Trainer (Logistic Regression)")
st.markdown(
    "Upload your `all_features.csv` (one row per sample; features columns + label column). "
    "The app trains a logistic regression, shows metrics, and lets you download the trained model (.pkl)."
)

# -------------------------
# File uploader / load data
# -------------------------
uploaded = st.file_uploader("Upload CSV file with features (CSV)", type=["csv"])
use_example = st.checkbox("Use example synthetic features (demo)", value=False)

if uploaded is None and not use_example:
    st.info("Upload a CSV or tick 'Use example synthetic features' to try the app.")
    st.stop()

if use_example:
    st.warning("Using synthetic example data (not your real data).")
    # create small synthetic dataset (2 classes)
    rng = np.random.RandomState(42)
    n = 300
    X1 = rng.normal(loc=0.0, scale=1.0, size=(n, 6))
    X2 = rng.normal(loc=1.2, scale=1.1, size=(n, 6))
    X = np.vstack([X1, X2])
    df = pd.DataFrame(X, columns=["RMS", "PeakToPeak", "Kurtosis", "Skewness", "SpectralCentroid", "SpectralEntropy"])
    df["label"] = np.hstack([np.zeros(n), np.ones(n)])
else:
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

st.subheader("Preview data")
st.dataframe(df.head(10))

# -------------------------
# Label selection & features
# -------------------------
st.sidebar.header("Training options")

cols = df.columns.tolist()
label_col = st.sidebar.selectbox("Select label column (target)", options=cols, index=(cols.index("label") if "label" in cols else 0))
st.sidebar.write(f"Using `{label_col}` as label/target.")

# Infer numeric columns for features excluding label
feature_cols = [c for c in cols if c != label_col and pd.api.types.is_numeric_dtype(df[c])]
if len(feature_cols) == 0:
    st.error("No numeric feature columns detected. Ensure your CSV has numeric features (and a numeric label).")
    st.stop()

chosen_features = st.sidebar.multiselect("Select features to use (ctrl/shift-click to multi-select)", options=feature_cols, default=feature_cols)
if len(chosen_features) == 0:
    st.error("Pick at least one feature column.")
    st.stop()

test_size = st.sidebar.slider("Test set fraction", 0.05, 0.5, value=0.2, step=0.05)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)
penalty = st.sidebar.selectbox("Logistic penalty", options=["l2", "none"], index=0)
C_val = st.sidebar.number_input("Inverse regularization C (larger = less reg)", min_value=0.0001, value=1.0, step=0.1, format="%.4f")
max_iter = st.sidebar.number_input("Max iterations", min_value=50, max_value=10000, value=1000, step=50)

# -------------------------
# Prepare X, y
# -------------------------
# Ensure label is numeric binary
y_raw = df[label_col]
if y_raw.dtype == object:
    # try to map to numeric
    try:
        y = pd.factorize(y_raw)[0]
    except Exception as e:
        st.error("Label column is non-numeric and could not be factorized automatically.")
        st.stop()
else:
    y = y_raw.values

X = df[chosen_features].values

# Check binary labels
unique_labels = np.unique(y)
if unique_labels.shape[0] != 2:
    st.warning(f"Detected labels: {unique_labels}. Logistic regression will still train, but app expects a binary target.")
    
# -------------------------
# Train / Validate
# -------------------------
if st.button("Train model"):
    with st.spinner("Training logistic regression..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state), stratify=(y if len(unique_labels)==2 else None)
        )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(penalty=penalty if penalty!="none" else "none", C=float(C_val), max_iter=int(max_iter), solver="lbfgs" if penalty!="none" else "saga"))
        ])
        # If penalty == 'none', sklearn >=1.1 requires solver='saga' with penalty='none'; handle exceptions later.

        # fit
        try:
            pipeline.fit(X_train, y_train)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

        # predict + metrics
        y_pred = pipeline.predict(X_test)
        if hasattr(pipeline, "predict_proba"):
            y_score = pipeline.predict_proba(X_test)[:, 1]
        else:
            # fallback: decision_function
            try:
                y_score = pipeline.decision_function(X_test)
            except Exception:
                y_score = None

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=False)

        st.success(f"Training completed — accuracy: {acc:.4f}")
        st.write("## Classification report")
        st.text(report)

        st.write("## Confusion matrix")
        fig, ax = plt.subplots(figsize=(4,3))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, int(val), ha='center', va='center', color='black')
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)

        # ROC plot if possible
        if y_score is not None and len(np.unique(y_test)) == 2:
            fpr, tpr, thr = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            st.write(f"ROC AUC: {roc_auc:.4f}")
            fig2, ax2 = plt.subplots(figsize=(5,4))
            ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax2.plot([0,1],[0,1], linestyle="--", color="gray")
            ax2.set_xlabel("FPR")
            ax2.set_ylabel("TPR")
            ax2.set_title("ROC Curve")
            ax2.legend()
            st.pyplot(fig2)
        else:
            st.info("ROC/AUC not shown (predict_proba/decision_function unavailable or non-binary labels).")

        # -------------------------
        # Export model (.pkl)
        # -------------------------
        # Prepare pickle bytes
        buffer = BytesIO()
        joblib.dump(pipeline, buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()

        # Provide download button
        st.write("## Download trained model")
        pkl_name = st.text_input("Filename for download (must end with .pkl)", value="damage_model.pkl")
        if not pkl_name.endswith(".pkl"):
            pkl_name = pkl_name + ".pkl"

        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pkl_name}">Click here to download the trained model (.pkl)</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Also show how to use the model in code
        st.write("---")
        st.write("### Example: load & use the model in Python")
        st.code(f"""import joblib
model = joblib.load("{pkl_name}")
# X_new: 2D numpy array with same feature order
y_pred = model.predict(X_new)
y_prob = model.predict_proba(X_new)[:,1]  # if available""")
