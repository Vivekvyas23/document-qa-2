# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import base64
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="AE Features -> Logistic Regression", layout="wide")
st.title("AE Features → Logistic Regression Trainer")

st.markdown("""
Upload your `ae_features_all.csv` (or place it in the app folder).  
The app will:
- Drop `File_id` and `Label` from features (you can change these),
- Convert `Label` to numeric (Healthy -> 0, others -> 1),
- Train a Logistic Regression,
- Show metrics and confusion matrix,
- Let you download the trained `.pkl`.
""")

# -------------------------
# File input
# -------------------------
uploaded = st.file_uploader("Upload ae_features_all.csv (or leave empty to load local file named 'ae_features_all.csv')", type=["csv"])
use_local = False
if uploaded is None:
    st.info("No uploaded file. Will try to load 'ae_features_all.csv' from the app folder if present.")
    try:
        data = pd.read_csv("ae_features_all.csv")
        use_local = True
        st.success("Loaded local file 'ae_features_all.csv'.")
    except FileNotFoundError:
        st.warning("Local file not found. Upload a CSV to proceed.")
        st.stop()
else:
    try:
        data = pd.read_csv(uploaded)
        st.success("Uploaded CSV loaded.")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()

st.subheader("Preview of data")
st.dataframe(data.head())

# -------------------------
# Options / preprocessing
# -------------------------
st.sidebar.header("Preprocessing & training options")

drop_fileid = st.sidebar.checkbox("Drop 'File_id' column if present", value=True)
drop_band = st.sidebar.checkbox("Drop 'Band' column if present", value=False)
label_col = st.sidebar.text_input("Label column name", value="Label")
label_mapping_health = st.sidebar.text_input("Value representing 'Healthy' in label column", value="Healthy")
apply_scaling = st.sidebar.checkbox("Apply StandardScaler (fit on train set)", value=True)
test_size = st.sidebar.slider("Test set fraction", min_value=0.05, max_value=0.5, value=0.3, step=0.05)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

# -------------------------
# Prepare X and y (mimic your script)
# -------------------------
df = data.copy()

# Drop specified columns if present
cols_to_drop = []
if drop_fileid and 'File_id' in df.columns:
    cols_to_drop.append('File_id')
if drop_band and 'Band' in df.columns:
    cols_to_drop.append('Band')
# Always keep label_col for now
# Drop them later from X
if len(cols_to_drop) > 0:
    st.sidebar.write(f"Dropping columns: {cols_to_drop}")

# Check label column exists
if label_col not in df.columns:
    st.error(f"Label column '{label_col}' not found in data. Please correct the label column name in the sidebar.")
    st.stop()

# Map labels to 0/1
def map_label(val):
    try:
        # string comparison if provided as string
        if isinstance(val, str):
            return 0 if val.strip() == label_mapping_health else 1
        else:
            # numeric or other: treat as 0 if equals mapping string cast, else 1
            return 0 if str(val) == str(label_mapping_health) else 1
    except Exception:
        return 1

y = df[label_col].apply(map_label).astype(int)

# Build X by dropping label and optional columns
X = df.drop(columns=[label_col] + [c for c in cols_to_drop if c in df.columns], errors='ignore')

# Remove non-numeric columns from X (or attempt to convert)
non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
if non_numeric:
    st.warning(f"Found non-numeric columns in features. They will be dropped: {non_numeric}")
    X = X.drop(columns=non_numeric)

if X.shape[1] == 0:
    st.error("No numeric feature columns remain after dropping. Cannot train.")
    st.stop()

st.write("Feature columns used:", X.columns.tolist())

# -------------------------
# Train button
# -------------------------
if st.button("Train Logistic Regression (as in your code)"):
    # Train/test split
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=float(test_size), random_state=int(random_state), stratify=y.values
        )
    except Exception as e:
        st.error(f"train_test_split failed: {e}")
        st.stop()

    # Optional scaling (mimics good practice; can be turned off to match original code exactly)
    scaler = None
    if apply_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test

    # Train logistic regression
    try:
        model = LogisticRegression(random_state=int(random_state), max_iter=1000)
        model.fit(X_train_scaled, y_train)
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = None
    try:
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    except Exception:
        y_prob = None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.subheader("=== Logistic Regression Model Evaluation ===")
    st.write(f"**Accuracy:** {acc:.4f}")
    st.write("**Classification Report:**")
    st.text(report)

    st.write("**Confusion Matrix:**")
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
        ax2.set_xlabel("FPR")
        ax2.set_ylabel("TPR")
        ax2.set_title("ROC Curve")
        ax2.legend()
        st.pyplot(fig2)
    else:
        st.info("ROC/AUC not available (predict_proba missing or non-binary labels).")

    # Offer model download (save scaler, model, feature names)
    export_obj = {
        "model": model,
        "scaler": scaler,
        "feature_columns": list(X.columns)
    }

    buffer = BytesIO()
    joblib.dump(export_obj, buffer)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()

    st.subheader("Download trained model (.pkl)")
    suggested_name = "logistic_regression_model.pkl"
    pkl_name = st.text_input("Filename for download", value=suggested_name)
    if not pkl_name.endswith(".pkl"):
        pkl_name = pkl_name + ".pkl"

    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{pkl_name}">Click here to download the trained model (.pkl)</a>'
    st.markdown(href, unsafe_allow_html=True)

    st.info("Saved object contains keys: 'model' (sklearn LogisticRegression), 'scaler' (StandardScaler or None), 'feature_columns' (list).")
