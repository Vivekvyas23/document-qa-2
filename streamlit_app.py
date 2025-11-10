# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import joblib
import base64

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Damage-Detection Trainer (no Pipeline)", layout="wide")
st.title("🔧 Damage-Detection Trainer (Logistic Regression, no Pipeline)")

st.markdown("""
Upload a CSV containing numeric feature columns and a label column.
This app trains a `StandardScaler` (separately) and a `LogisticRegression`,
shows metrics (accuracy, confusion matrix, ROC), and provides a downloadable `.pkl`
that contains the scaler, model, and the feature order.
""")

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
# Sidebar - options
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
y_raw = df[label_col]
# Factorize non-numeric labels
if y_raw.dtype == object or not np.issubdtype(y_raw.dtype, np.number):
    y = pd.factorize(y_raw)[0]
else:
    y = y_raw.values

X_df = df[chosen_features].copy()
X = X_df.values

unique_labels = np.unique(y)
if unique_labels.shape[0] != 2:
    st.warning(f"Detected labels: {unique_labels}. Logistic regression expects binary target but will still train.")

# -------------------------
# Train / Validate
# -------------------------
if st.button("Train model"):
    with st.spinner("Training logistic regression (separate StandardScaler + LogisticRegression)..."):
        try:
            stratify_arg = y if len(unique_labels) == 2 else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=int(random_state), stratify=stratify_arg
            )
        except Exception as e:
            st.error(f"Failed during train_test_split: {e}")
            st.stop()

        # StandardScaler (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Logistic Regression (no sklearn Pipeline)
        solver_choice = "lbfgs" if penalty == "l2" else "saga"
        try:
            model = LogisticRegression(penalty=penalty, C=float(C_val), max_iter=int(max_iter), solver=solver_choice)
        except Exception:
            # fallback if 'none' isn't accepted
            model = LogisticRegression(penalty="l2", C=float(C_val), max_iter=int(max_iter), solver="lbfgs")

        try:
            model.fit(X_train_scaled, y_train)
        except Exception as e:
            st.error(f"Training failed: {e}")
            st.stop()

        # Predictions & Scores
        y_pred = model.predict(X_test_scaled)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test_scaled)[:, 1]
        else:
            try:
                y_score = model.decision_function(X_test_scaled)
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

        # ROC plot
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

        # Show model coefficients (optional)
        coef_btn = st.checkbox("Show model coefficients", value=False)
        if coef_btn:
            coefs = model.coef_.ravel()
            coef_df = pd.DataFrame({"feature": chosen_features, "coefficient": coefs})
            coef_df = coef_df.sort_values("coefficient", key=lambda col: np.abs(col), ascending=False)
            st.write(coef_df)

        # -------------------------
        # Export model (.pkl) - save scaler, model, feature list in a dict
        # -------------------------
        export_obj = {
            "scaler": scaler,
            "model": model,
            "features": chosen_features
        }

        buffer = BytesIO()
        joblib.dump(export_obj, buffer)
        buffer.seek(0)
        b64 = base64.b64encode(buffer.read()).decode()

        st.write("## Download trained model")
        pkl_name = st.text_input("Filename for download (must end with .pkl)", value="damage_model.pkl")
        if not pkl_name.endswith(".pkl"):
            pkl_name = pkl_name + ".pkl"

        href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pkl_name}">Click here to download the trained model (.pkl)</a>'
        st.markdown(href, unsafe_allow_html=True)

        # Show usage snippet
        st.write("---")
        st.write("### Example: load & use the saved object")
        st.code(f"""import joblib
obj = joblib.load("{pkl_name}")   # dict with keys: 'scaler','model','features'
scaler = obj['scaler']
model = obj['model']
features = obj['features']
# X_new must be a 2D array with columns in the same order as 'features'
X_new_scaled = scaler.transform(X_new)
y_pred = model.predict(X_new_scaled)
y_prob = model.predict_proba(X_new_scaled)[:,1]  # if available""")
