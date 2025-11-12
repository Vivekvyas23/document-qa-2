# streamlit_app_exact.py
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from io import BytesIO
import base64

st.set_page_config(page_title="Exact: Train Logistic from CSV", layout="wide")
st.title("Exact port of your script → Streamlit (no logic changed)")

st.markdown("""
This app reproduces your script exactly:
- Reads CSV (tries uploaded file first; if none provided it will attempt to read `/content/new_feature.csv`)
- Drops `file` and `label` from X
- Maps label with `lambda x: 0 if x == 'Healthy' else 1` (case-sensitive)
- Uses train_test_split(test_size=0.3, random_state=42)
- Trains `LogisticRegression(random_state=42)` with default params
- Prints accuracy, classification report, confusion matrix
- Saves model with `joblib.dump(model, 'logistic_regression_model.pkl')` and provides download link
""")

# --- File input ---
uploaded = st.file_uploader("Upload CSV (same format as '/content/new_feature.csv')", type=["csv"])
if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        st.success("Uploaded CSV loaded.")
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        st.stop()
else:
    # Try to load the exact path you used
    try:
        data = pd.read_csv('/content/new_feature.csv')
        st.info("Loaded '/content/new_feature.csv' from disk.")
    except Exception:
        st.warning("No uploaded file and '/content/new_feature.csv' not found. Please upload your CSV to proceed.")
        st.stop()

st.subheader("Preview (first 8 rows)")
st.dataframe(data.head(8))

# --- EXACT original script logic starts here ---
# (kept variable names and code structure as in your snippet)

# Prepare X and y as in your code
try:
    X = data.drop(['file', 'label'], axis=1)
except Exception as e:
    st.error(f"Error dropping ['file','label'] from dataframe: {e}")
    st.stop()

# label mapping exactly as in your code (case-sensitive)
y = data['label'].apply(lambda x: 0 if x == 'Healthy' else 1)

# Train/test split (exact params)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train LogisticRegression exactly as in your code
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Predictions and evaluation (same as your script)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Print outputs (matching your print style)
st.write("=== Logistic Regression Model Evaluation ===")
st.write(f"Accuracy: {accuracy:.4f}")

st.write("Classification Report:")
st.text(report)

st.write("Confusion Matrix:")
st.write(conf_matrix)

# Save model exactly as in your script
model_filename = 'logistic_regression_model.pkl'
joblib.dump(model, model_filename)
st.success(f"Model saved successfully to {model_filename}")

# Provide download link for the saved file
with open(model_filename, "rb") as f:
    bytes_data = f.read()
b64 = base64.b64encode(bytes_data).decode()
href = f'<a href="data:application/octet-stream;base64,{b64}" download="{model_filename}">Download {model_filename}</a>'
st.markdown(href, unsafe_allow_html=True)

st.info("Note: This app preserves your original script logic exactly (label mapping is case-sensitive). If you want label cleaning or numeric coercion, request it separately.")
