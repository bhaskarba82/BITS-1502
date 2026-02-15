
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="Adult Income ML Dashboard", layout="wide")

st.title("💼 Adult Income Classification Dashboard")
st.markdown("Predict whether income exceeds **$50K per year** using multiple ML models.")

# ==========================================
# Load Models & Preprocessing Artifacts
# ==========================================

@st.cache_resource
def load_artifacts():
    base_path = "model"

    models = {
        "Logistic Regression": joblib.load(os.path.join(base_path, "logistic_regression.pkl")),
        "Decision Tree": joblib.load(os.path.join(base_path, "decision_tree.pkl")),
        "KNN": joblib.load(os.path.join(base_path, "knn.pkl")),
        "Naive Bayes": joblib.load(os.path.join(base_path, "naive_bayes.pkl")),
        "Random Forest": joblib.load(os.path.join(base_path, "random_forest.pkl")),
        "XGBoost": joblib.load(os.path.join(base_path, "xgboost.pkl"))
    }

    scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(base_path, "features.pkl"))

    return models, scaler, feature_columns

models, scaler, feature_columns = load_artifacts()

# ==========================================
# Model Selection
# ==========================================

model_option = st.selectbox("Select Model", list(models.keys()))
selected_model = models[model_option]

# ==========================================
# Evaluation Metrics Display
# ==========================================

metrics = {
    "Logistic Regression": [0.852975, 0.907738, 0.756994, 0.618954, 0.681050, 0.591786],
    "Decision Tree": [0.812531, 0.757259, 0.626667, 0.645098, 0.635749, 0.509668],
    "KNN": [0.813194, 0.831738, 0.653465, 0.560784, 0.603588, 0.484737],
    "Naive Bayes": [0.450191, 0.694940, 0.310899, 0.960131, 0.469704, 0.250291],
    "Random Forest": [0.850820, 0.902255, 0.736486, 0.641176, 0.685535, 0.590791],
    "XGBoost": [0.870214, 0.927999, 0.781038, 0.678431, 0.726128, 0.644366]
}

st.subheader("📊 Model Evaluation Metrics")
vals = metrics[model_option]

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", vals[0])
col1.metric("AUC", vals[1])
col2.metric("Precision", vals[2])
col2.metric("Recall", vals[3])
col3.metric("F1 Score", vals[4])
col3.metric("MCC", vals[5])

# ==========================================
# Preprocessing Function
# ==========================================

def preprocess_input(df):
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    df_scaled = scaler.transform(df)
    return df_scaled

# ==========================================
# CSV Upload Section
# ==========================================

st.header("📂 Upload Test Dataset (CSV)")
uploaded_file = st.file_uploader("Upload Test CSV File (must match training features)", type=["csv"])

if uploaded_file is not None:
    test_data = pd.read_csv(uploaded_file)

    if "income" in test_data.columns:
        y_true = test_data["income"]
        X_test = test_data.drop("income", axis=1)
    else:
        y_true = None
        X_test = test_data

    try:
        processed_data = preprocess_input(X_test)
        predictions = selected_model.predict(processed_data)

        test_data["Prediction"] = predictions
        test_data["Prediction"] = test_data["Prediction"].map(
            {0: "Income <= 50K", 1: "Income > 50K"}
        )

        st.success("Predictions Completed Successfully ✅")
        st.dataframe(test_data)

        # Confusion Matrix (Single Plot)
        if y_true is not None:
            st.subheader("📉 Confusion Matrix")
            cm = confusion_matrix(y_true, predictions)

            fig = plt.figure()
            plt.imshow(cm)
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.colorbar()
            st.pyplot(fig)

    except Exception as e:
        st.error("Error processing uploaded file.")
        st.write(e)

# ==========================================
# Feature Importance (Single Plot)
# ==========================================

st.header("📊 Feature Importance")

if hasattr(selected_model, "feature_importances_"):
    importances = selected_model.feature_importances_
    feature_df = pd.DataFrame({
        "Feature": feature_columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(15)

    fig = plt.figure()
    plt.barh(feature_df["Feature"], feature_df["Importance"])
    plt.gca().invert_yaxis()
    plt.title("Top 15 Important Features")
    st.pyplot(fig)

else:
    st.info("Feature importance not available for this model.")
