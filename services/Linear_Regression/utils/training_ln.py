import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline


def training(df):
    # Chia dữ liệu thành đầu vào (X) và nhãn (y)
    X = df.drop(columns=['Survived'], axis=1)
    y = df['Survived']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [
                'Age', 'Fare', 'Pclass', 'SibSp', 'Parch']),
            ('cat', OneHotEncoder(), ['Embarked', 'Sex'])
        ]
    )

    st.header("Cài đặt tham số")

    # Chọn tỷ lệ chia dữ liệu
    train_ratio = st.slider("Train ratio (%)", 50, 80, 70)
    valid_ratio = st.slider("Validation ratio (%)", 5, 30, 15)
    test_ratio = 100 - train_ratio - valid_ratio

    # if test_ratio <= 0:
    #     st.error("Tổng tỷ lệ phải bằng 100%!")
    #     return

    random_state = st.slider("Random state", 0, 100, 42)
    k_fold = st.slider("Số k trong Cross Validation", 2, 10, 5)

    # test_ratio = st.slider("Test set ratio (%)", 10, 20, 15)
    # k_fold = st.slider("Số k trong Cross Validation", 2, 10, 5)
    # random_state = st.slider("Random state", 0, 100, 42)

    # Chọn model
    # model_type = st.selectbox(
    #     "Chọn mô hình", ["Multiple Regression", "Polynomial Regression"]
    # )

    # degree = 2
    # if model_type == "Polynomial Regression":
    #     degree = st.slider("Bậc đa thức", 2, 5, 2)

    # if model_type == "Polynomial Regression":
    #     model = Pipeline([
    #         ('preprocessor', preprocessor),
    #         ('poly', PolynomialFeatures(degree=degree)),
    #         ('regressor', LinearRegression())
    #     ])
    # else:
    #     model = Pipeline([
    #         ('preprocessor', preprocessor),
    #         ('regressor', LinearRegression())
    #     ])

    # Thực hiện chia dữ liệu
    # # Initialize K-Fold
    # kf = KFold(n_splits=k_fold, shuffle=True, random_state=random_state)

    # # Initialize lists to store scores
    # rmse_scores = []
    # r2_scores = []
    # fold_metrics = []
    # # Perform k-fold cross-validation
    # for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    #     # Split data for this fold
    #     X_train_fold = X.iloc[train_idx]
    #     X_val_fold = X.iloc[val_idx]
    #     y_train_fold = y.iloc[train_idx]
    #     y_val_fold = y.iloc[val_idx]

    #     # Train model on this fold
    #     pipeline.fit(X_train_fold, y_train_fold)

    #     # Make predictions on validation fold
    #     y_val_pred = pipeline.predict(X_val_fold)

    #     # Calculate metrics for this fold
    #     rmse = mean_squared_error(y_val_fold, y_val_pred, squared=False)
    #     r2 = r2_score(y_val_fold, y_val_pred)

    #     rmse_scores.append(rmse)
    #     r2_scores.append(r2)

    #     # Store fold metrics for logging
    #     fold_metrics.append({
    #         f"fold_{fold}_rmse": rmse,
    #         f"fold_{fold}_r2": r2
    #     })
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_ratio/100, random_state=random_state
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=valid_ratio/(train_ratio + valid_ratio), random_state=random_state
    )

    # Chọn model
    model_type = st.selectbox(
        "Chọn mô hình", ["Multiple Regression", "Polynomial Regression"]
    )

    degree = 2
    if model_type == "Polynomial Regression":
        degree = st.slider("Bậc đa thức", 2, 5, 2)

    # Huấn luyện và tracking với MLflow
    if st.button("Train Model"):
        with mlflow.start_run():
            mlflow.log_params({
                "model_type": model_type,
                "train_ratio": train_ratio,
                "valid_ratio": valid_ratio,
                "test_ratio": test_ratio,
                "degree": degree if model_type == "Polynomial Regression" else None,
                "k_fold": k_fold
            })

            # Tạo pipeline phù hợp với loại mô hình
            if model_type == "Polynomial Regression":
                model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('poly', PolynomialFeatures(degree=degree)),
                    ('regressor', LinearRegression())
                ])
            else:
                model = Pipeline([
                    ('preprocessor', preprocessor),
                    ('regressor', LinearRegression())
                ])

            # Cross-validation
            cv_rmse_scores = - \
                cross_val_score(model, X_train, y_train, cv=k_fold,
                                scoring='neg_mean_squared_error')
            cv_r2_scores = cross_val_score(
                model, X_train, y_train, cv=k_fold, scoring='r2')

            # Huấn luyện model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Log metrics
            mlflow.log_metrics({
                "cv_mse_mean": np.mean(cv_rmse_scores),
                "cv_r2_mean": np.mean(cv_r2_scores),
                "test_mse": mean_squared_error(y_test, y_pred, squared=False),
                "test_r2": r2_score(y_test, y_pred)
            })

            # Hiển thị kết quả
            st.subheader("Kết quả huấn luyện")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cross-Validation MSE",
                          f"{np.mean(cv_rmse_scores):.4f}")
                st.metric("Test R²", f"{r2_score(y_test, y_pred):.4f}")
            with col2:
                st.metric("Cross-Validation R²",
                          f"{np.mean(cv_r2_scores):.4f}")
                st.metric(
                    "Test MSE", f"{mean_squared_error(y_test, y_pred, squared=False):.4f}")

            # Log model
            mlflow.sklearn.log_model(model, "model")

            st.success(
                f"Model trained successfully! Run ID: {mlflow.active_run().info.run_id}")
