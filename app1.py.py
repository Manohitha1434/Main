# 1. Imports and Load Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
import joblib
import os

warnings.filterwarnings("ignore")

# Streamlit Title
st.title("Prognosis Prediction App")

# File Uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV file)", type=["csv"])
if uploaded_file is not None:
    # Read dataset
    data = pd.read_csv(uploaded_file, sep=";")
    
    # 2. Data Cleaning
    st.subheader("Data Cleaning")
    # Drop ID column
    if "ID" in data.columns:
        data = data.drop("ID", axis=1)
    # Check for missing values
    st.write("Missing Values:", data.isnull().sum())
    # Convert 'prognosis' to numeric if it's not already
    if data['prognosis'].dtype == 'object':
        data['prognosis'] = data['prognosis'].map({'no_retinopathy': 0, 'retinopathy': 1})
    st.write("Data Types:", data.dtypes)

    # 3. Exploratory Data Analysis (EDA)
    st.subheader("Exploratory Data Analysis (EDA)")
    # Age Distribution
    st.write("Age Distribution:")
    fig, ax = plt.subplots()
    sns.histplot(data['age'], kde=True, ax=ax)
    st.pyplot(fig)

    # Cholesterol vs Prognosis
    st.write("Cholesterol vs Prognosis:")
    fig, ax = plt.subplots()
    sns.boxplot(x='prognosis', y='cholesterol', data=data, ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.write("Feature Correlation:")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # 4. Preprocessing
    st.subheader("Data Preprocessing")
    X = data.drop('prognosis', axis=1)
    y = data['prognosis']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # 5. Model Building and Evaluation
    st.subheader("Model Building and Evaluation")
    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_pred)
        results[name] = [acc, prec, rec, roc]
        st.write(f"{name} Results:")
        st.text(classification_report(y_test, y_pred))

    # 6. Model Comparison
    st.subheader("Model Comparison")
    results_df = pd.DataFrame(results, index=['Accuracy', 'Precision', 'Recall', 'ROC-AUC'])
    st.write(results_df)

    # 7. Save the Best Model
    st.subheader("Save the Best Model")
    best_model = RandomForestClassifier()
    best_model.fit(X_train, y_train)
    
    # Create the 'models' directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
    joblib.dump(best_model, 'models/best_model.pkl')
    st.write("Best model saved successfully!")
else:
    st.warning("Please upload a dataset to proceed.")