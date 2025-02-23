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

from services.Linear_Regression.utils.preprocess import preprocess_data


def training(data):
    st.write("## 📊 Chia dữ liệu (Training - Validation - Testing)")

    # Kiểm tra nếu `target_column` chưa tồn tại
    if "target_column" not in st.session_state or st.session_state.target_column not in data.columns:
        # Mặc định chọn cột đầu tiên
        st.session_state.target_column = data.columns[0]

    selected_label = st.selectbox("Chọn cột dự đoán", data.columns,
                                  index=data.columns.get_loc(st.session_state.target_column) if st.session_state.target_column else 0)

    X = data.drop(columns=[selected_label], axis=1)
    y = data[selected_label]

    # Kiểm tra `st.session_state.data`
    if "data" in st.session_state:
        data = st.session_state.data

    test_size = st.slider("Chọn % testing", 10, 50, 20)
    val_size = st.slider("Chọn % validation", 0, 50, 15)
    remaining_size = 100 - test_size

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42)

    # Chia tập Train - Validation bằng Cross Validation (KFold)
    k = st.slider("Chọn số k trong Cross Validation", 2, 10, 5)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    mse_list = []

    for train_index, val_index in kf.split(X_train_full):
        X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
        y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]

    st.write("### Tỷ lệ chia dữ liệu")
    table_size = pd.DataFrame({
        'Dataset': ['Train', 'Validation', 'Test'],
        'Kích thước (%)': [remaining_size - val_size, val_size, test_size],
        'Số lượng mẫu': [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
    })
    st.write(table_size)
    st.write(f" - Số fold Cross Validation: {k}")

    st.session_state.X_train = X_train
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_test = y_test
    st.session_state.X_train_full = X_train_full
    st.session_state.y_train_full = y_train_full

    st.markdown("---")
    st.write("## ⚙️ Huấn luyện mô hình")

    model_option = st.selectbox(
        "Chọn mô hình", ["Multiple Regression", "Polynomial Regression"])

    if model_option == "Polynomial Regression":
        degree = st.slider("Chọn bậc của Polynomial Regression", 2, 5, 3)

    if st.button("Huấn luyện mô hình"):
        fold_mse = []

        for train_index, val_index in kf.split(X_train_full):
            X_train, X_val = X_train_full.iloc[train_index], X_train_full.iloc[val_index]
            y_train, y_val = y_train_full.iloc[train_index], y_train_full.iloc[val_index]

            if model_option == "Multiple Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)

            elif model_option == "Polynomial Regression":
                poly = PolynomialFeatures(degree=degree)
                X_train_poly = poly.fit_transform(X_train)
                X_val_poly = poly.transform(X_val)

                model = LinearRegression()
                model.fit(X_train_poly, y_train)
                y_pred = model.predict(X_val_poly)

            mse = mean_squared_error(y_val, y_pred)
            fold_mse.append(mse)

        col1, col2 = st.columns(2)
        # Plot MSE for each fold
        with col1:
            st.write("###### Kết quả Cross Validation qua từng fold")
            results_df = pd.DataFrame({
                'MSE': fold_mse
            })
            st.write(results_df)
        with col2:
            fig, ax = plt.subplots()
            ax.plot(range(1, k + 1), fold_mse,
                    marker='o', linestyle='-', color='b')
            ax.set_xlabel('Fold')
            ax.set_ylabel('MSE')
            ax.set_title('Cross Validation MSE qua từng fold ')
            st.pyplot(fig)

        # Lưu mô hình vào session_state
        st.session_state.model = model

        # ✅ Cuối cùng, đánh giá mô hình trên tập TEST (sau khi đã chọn mô hình tốt nhất từ Cross Validation)
        if model_option == "Multiple Regression":
            y_test_pred = model.predict(X_test)
        else:
            X_test_poly = poly.transform(X_test)
            y_test_pred = model.predict(X_test_poly)

        test_mse = mean_squared_error(y_test, y_test_pred)
        avg_mse = np.mean(fold_mse)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("**MSE trung bình qua từng fold:**", f"{avg_mse:.4f}")
        with col2:
            st.metric("**MSE trên tập Test:**", f"{test_mse:.4f}")


# def training(df):
#     # Chia dữ liệu thành đầu vào (X) và nhãn (y)
#     X = df.drop(columns=['Survived'], axis=1)
#     y = df['Survived']

#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), [
#                 'Age', 'Fare', 'Pclass', 'SibSp', 'Parch']),
#             ('cat', OneHotEncoder(), ['Embarked', 'Sex'])
#         ]
#     )

#     st.header("Cài đặt tham số")

#     # Chọn tỷ lệ chia dữ liệu
#     train_ratio = st.slider("Train ratio (%)", 50, 80, 70)
#     valid_ratio = st.slider("Validation ratio (%)", 5, 30, 15)
#     test_ratio = 100 - train_ratio - valid_ratio

#     if test_ratio <= 0:
#         st.error("Tổng tỷ lệ phải bằng 100%!")
#         return

#     random_state = st.slider("Random state", 0, 100, 42)
#     k_fold = st.slider("Số k trong Cross Validation", 2, 10, 5)

#     X_train_val, X_test, y_train_val, y_test = train_test_split(
#         X, y, test_size=test_ratio/100, random_state=random_state
#     )

#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train_val, y_train_val, test_size=valid_ratio/(train_ratio + valid_ratio), random_state=random_state
#     )

#     # Chọn model
#     model_type = st.selectbox(
#         "Chọn mô hình", ["Multiple Regression", "Polynomial Regression"]
#     )

#     degree = 2
#     if model_type == "Polynomial Regression":
#         degree = st.slider("Bậc đa thức", 2, 5, 2)

#     # Huấn luyện và tracking với MLflow
#     if st.button("Train Model"):
#         with mlflow.start_run():
#             mlflow.log_params({
#                 "model_type": model_type,
#                 "train_ratio": train_ratio,
#                 "valid_ratio": valid_ratio,
#                 "test_ratio": test_ratio,
#                 "degree": degree if model_type == "Polynomial Regression" else None,
#                 "k_fold": k_fold
#             })

#             # Tạo pipeline phù hợp với loại mô hình
#             if model_type == "Polynomial Regression":
#                 model = Pipeline([
#                     ('preprocessor', preprocessor),
#                     ('poly', PolynomialFeatures(degree=degree)),
#                     ('regressor', LinearRegression())
#                 ])
#             else:
#                 model = Pipeline([
#                     ('preprocessor', preprocessor),
#                     ('regressor', LinearRegression())
#                 ])

#             # Cross-validation
#             cv_rmse_scores = - \
#                 cross_val_score(model, X_train, y_train, cv=k_fold,
#                                 scoring='neg_mean_squared_error')
#             cv_r2_scores = cross_val_score(
#                 model, X_train, y_train, cv=k_fold, scoring='r2')

#             # Huấn luyện model
#             model.fit(X_train, y_train)
#             y_pred = model.predict(X_test)

#             # Log metrics
#             mlflow.log_metrics({
#                 "cv_mse_mean": np.mean(cv_rmse_scores),
#                 "cv_r2_mean": np.mean(cv_r2_scores),
#                 "test_mse": mean_squared_error(y_test, y_pred, squared=False),
#                 "test_r2": r2_score(y_test, y_pred)
#             })

#             # Hiển thị kết quả
#             st.subheader("Kết quả huấn luyện")
#             col1, col2 = st.columns(2)
#             with col1:
#                 st.metric("Cross-Validation MSE",
#                           f"{np.mean(cv_rmse_scores):.4f}")
#                 st.metric("Test R²", f"{r2_score(y_test, y_pred):.4f}")
#             with col2:
#                 st.metric("Cross-Validation R²",
#                           f"{np.mean(cv_r2_scores):.4f}")
#                 st.metric(
#                     "Test MSE", f"{mean_squared_error(y_test, y_pred, squared=False):.4f}")

#             # Log model
#             mlflow.sklearn.log_model(model, "model")

#             st.success(
#                 f"Model trained successfully! Run ID: {mlflow.active_run().info.run_id}")
