# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.train import train_models
from src.evaluate import evaluate_model


st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.write("Milestone 1 - Machine Learning Based Churn Prediction")


# ---------------------------------------------
# Sidebar - Upload Dataset
# ---------------------------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload Clean Telco CSV",
    type=["csv"]
)


# ---------------------------------------------
# Main App
# ---------------------------------------------
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())
    st.write("Dataset Shape:", df.shape)

    # Dataset validation
    if "Churn Value" not in df.columns:
        st.error("❌ Invalid dataset. Please upload the correct Telco Churn dataset.")
        st.stop()

    if st.button("🚀 Train Models"):

        with st.spinner("Training models..."):

            results = train_models(df)

            log_model = results["log_model"]
            tree_model = results["tree_model"]
            best_model = results["best_model"]
            best_model_name = results["best_model_name"]
            best_accuracy = results["best_accuracy"]
            X_test = results["X_test"]
            y_test = results["y_test"]

        st.success("Training Completed!")

        # ---------------------------------------------
        # Logistic Regression
        # ---------------------------------------------
        st.subheader("🔵 Logistic Regression Performance")

        log_metrics = evaluate_model(log_model, X_test, y_test)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", round(log_metrics["Accuracy"], 3))
        col2.metric("Precision", round(log_metrics["Precision"], 3))
        col3.metric("Recall", round(log_metrics["Recall"], 3))
        col4.metric("F1 Score", round(log_metrics["F1 Score"], 3))

        fig1, ax1 = plt.subplots()
        sns.heatmap(log_metrics["Confusion Matrix"], annot=True, fmt="d", cmap="Blues", ax=ax1)
        ax1.set_title("Logistic Regression Confusion Matrix")
        ax1.set_xlabel("Predicted")
        ax1.set_ylabel("Actual")
        st.pyplot(fig1)


        # ---------------------------------------------
        # Decision Tree
        # ---------------------------------------------
        st.subheader("🟢 Decision Tree Performance")

        tree_metrics = evaluate_model(tree_model, X_test, y_test)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", round(tree_metrics["Accuracy"], 3))
        col2.metric("Precision", round(tree_metrics["Precision"], 3))
        col3.metric("Recall", round(tree_metrics["Recall"], 3))
        col4.metric("F1 Score", round(tree_metrics["F1 Score"], 3))

        fig2, ax2 = plt.subplots()
        sns.heatmap(tree_metrics["Confusion Matrix"], annot=True, fmt="d", cmap="Greens", ax=ax2)
        ax2.set_title("Decision Tree Confusion Matrix")
        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")
        st.pyplot(fig2)


        # ---------------------------------------------
        # Feature Importance
        # ---------------------------------------------
        st.subheader("📌 Top 10 Important Features (Decision Tree)")

        importances = tree_model.named_steps["classifier"].feature_importances_
        feature_names = tree_model.named_steps["preprocessor"].get_feature_names_out()

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(10)

        st.dataframe(importance_df)


        # ---------------------------------------------
        # Best Model
        # ---------------------------------------------
        st.subheader("🏆 Best Model")

        st.write(f"Best Model: **{best_model_name}**")
        st.write(f"Best Accuracy: **{round(best_accuracy, 3)}**")


        # ---------------------------------------------
        # Sample Probability
        # ---------------------------------------------
        st.subheader("📌 Sample Churn Probability (Best Model)")

        sample = X_test.iloc[[0]]
        probability = best_model.predict_proba(sample)[0][1]

        st.success(f"Predicted Churn Probability: {round(probability * 100, 2)} %")