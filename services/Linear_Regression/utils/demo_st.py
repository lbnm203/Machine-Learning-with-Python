import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score, mean_squared_error


from services.Linear_Regression.utils.preprocess import preprocess_data


def demo_app(X, y, model, model_type):
    
    model.fit(X, y)  # Đảm bảo mô hình được huấn luyện trước khi dự đoán

    st.header("Dự đoán sống sót")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Pclass", [1, 2, 3])
            age = st.number_input("Age", min_value=0, max_value=100, value=25)
            fare = st.number_input("Fare", min_value=0, value=50)
        with col2:
            sex = st.selectbox("Sex", ["male", "female"])
            embarked = st.selectbox("Embarked", ["C", "Q", "S"])
            sibsp = st.number_input("Sibsp", min_value=0, value=10)
            parch = st.number_input("Parch", min_value=0, value=1000)

        if st.form_submit_button("Dự đoán"):
            input_data = pd.DataFrame(
                [[pclass, sex, age, fare, embarked, sibsp, parch]], columns=['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'Sibsp', 'Parch'])
            input_data = preprocess_data(input_data)

            prediction = model.predict(input_data)[0]
            probability = f"{prediction*100:.2f}%" if model_type == "Multiple Regression" else ""

            result = "Survived" if prediction > 0.5 else "Not Survived"
            st.subheader(f"Kết quả: **{result}**")
            if probability:
                st.write(f"Xác suất sống sót: {probability}")
