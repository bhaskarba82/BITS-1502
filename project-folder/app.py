
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Adult Income ML Dashboard", layout="wide")

st.title("💼 Adult Income Classification Dashboard")

@st.cache_resource
def load_artifacts():
    base_path = "project-folder/model"

    models = {
        "Logistic Regression": joblib.load(f"{base_path}/logistic_regression.pkl"),
        "Decision Tree": joblib.load(f"{base_path}/decision_tree.pkl"),
        "KNN": joblib.load(f"{base_path}/knn.pkl"),
        "Naive Bayes": joblib.load(f"{base_path}/naive_bayes.pkl"),
        "Random Forest": joblib.load(f"{base_path}/random_forest.pkl"),
        "XGBoost": joblib.load(f"{base_path}/xgboost.pkl")
    }
    scaler = joblib.load(f"{base_path}/scaler.pkl")
    feature_columns = joblib.load(f"{base_path}/features.pkl")
    return models, scaler, feature_columns

models, scaler, feature_columns = load_artifacts()

model_option = st.selectbox("Select Model", list(models.keys()))
selected_model = models[model_option]

st.header("Upload CSV for Prediction")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = data.reindex(columns=feature_columns, fill_value=0)
    scaled = scaler.transform(data)
    predictions = selected_model.predict(scaled)
    data["Prediction"] = predictions
    st.success("Prediction Complete ✅")
    st.dataframe(data)
