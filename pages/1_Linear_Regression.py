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
from services.Linear_Regression.utils.org_data import visualize_org_data
from services.Linear_Regression.utils.training_ln import training
from services.Linear_Regression.utils.theory_ln import theory_linear
from services.Linear_Regression.utils.demo_st import demo_app
from services.Linear_Regression.utils.show_mlflow import show_experiment_selector

# Khởi tạo MLflow
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# # mlflow.create_experiment("Titanic_Survival_Prediction")
# mlflow.set_experiment("Titanic_Survival_Prediction")


def main():
    st.title("Titanic Survival Prediction with Linear Regression")

    uploaded_file = st.file_uploader(
        "📂 Chọn file dữ liệu (.csv hoặc .txt)", type=["csv", "txt"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file, delimiter=",")
        st.success("📂 Tải file thành công!")

        data_org, data_preprocess, theory, train_process, demo, mlflow_p = st.tabs(
            ["Dữ liệu gốc", "Dữ liệu được tiền xử lý", "Thông tin", "Huấn luyện", "Demo", "Mlflow Tracking"])

        # ------------- Show Data Origin ------------------
        with data_org:
            visualize_org_data(data)

        # ------------- Preprocess Data ------------------
        with data_preprocess:
            data = preprocess_data(data)

        # ------------- Theoretical Background ------------------
        with theory:
            theory_linear()

        # ------------- Training Linear Regression ------------------
        with train_process:
            training(data)

        # ------------- Demo Application ------------------
        with demo:
            demo_app(data)

        # ------------- Mlflow Tracking ------------------
        with mlflow_p:
            show_experiment_selector()


if __name__ == "__main__":
    main()
