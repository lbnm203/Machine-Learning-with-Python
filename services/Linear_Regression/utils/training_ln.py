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
import os
from mlflow.tracking import MlflowClient


def mlflow_input():
    st.title(" MLflow Tracking ")

    DAGSHUB_MLFLOW_URI = "https://dagshub.com/lbnm203/Machine_Learning_UI.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "lbnm203"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "0902d781e6c2b4adcd3cbf60e0f288a8085c5aab"

    mlflow.set_experiment("Linear_Regression")


def train_multiple_linear_regression(X_train, y_train, learning_rate=0.001, n_iterations=200):
    """Huấn luyện hồi quy tuyến tính bội bằng Gradient Descent."""

    # Chuyển đổi X_train, y_train sang NumPy array để tránh lỗi
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train,
                                                              (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)

    # Kiểm tra NaN hoặc Inf
    if np.isnan(X_train).any() or np.isnan(y_train).any():
        raise ValueError("Dữ liệu đầu vào chứa giá trị NaN!")
    if np.isinf(X_train).any() or np.isinf(y_train).any():
        raise ValueError("Dữ liệu đầu vào chứa giá trị vô cùng (Inf)!")

    # Chuẩn hóa dữ liệu để tránh tràn số
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # Lấy số lượng mẫu (m) và số lượng đặc trưng (n)
    m, n = X_train.shape
    # st.write(f"Số lượng mẫu (m): {m}, Số lượng đặc trưng (n): {n}")

    # Thêm cột bias (x0 = 1) vào X_train
    X_b = np.c_[np.ones((m, 1)), X_train]
    # st.write(f"Kích thước ma trận X_b: {X_b.shape}")

    # Khởi tạo trọng số ngẫu nhiên nhỏ
    w = np.random.randn(X_b.shape[1], 1) * 0.01
    # st.write(f"Trọng số ban đầu: {w.flatten()}")

    # Gradient Descent
    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(w) - y_train)

        # Kiểm tra xem gradients có NaN không
        # st.write(gradients)
        if np.isnan(gradients).any():
            raise ValueError(
                "Gradient chứa giá trị NaN! Hãy kiểm tra lại dữ liệu hoặc learning rate.")

        w -= learning_rate * gradients

    # st.success("✅ Huấn luyện hoàn tất!")
    # st.write(f"Trọng số cuối cùng: {w.flatten()}")
    return w


def train_polynomial_regression(X_train, y_train, degree=2, learning_rate=0.001, n_iterations=500):
    """Huấn luyện hồi quy đa thức **không có tương tác** bằng Gradient Descent."""

    # Chuyển dữ liệu sang NumPy array nếu là pandas DataFrame/Series
    X_train = X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train
    y_train = y_train.to_numpy().reshape(-1, 1) if isinstance(y_train,
                                                              (pd.Series, pd.DataFrame)) else y_train.reshape(-1, 1)

    # Tạo đặc trưng đa thức **chỉ thêm bậc cao, không có tương tác**
    X_poly = np.hstack([X_train] + [X_train**d for d in range(2, degree + 1)])
    # Chuẩn hóa dữ liệu để tránh tràn số
    scaler = StandardScaler()
    X_poly = scaler.fit_transform(X_poly)

    # Lấy số lượng mẫu (m) và số lượng đặc trưng (n)
    m, n = X_poly.shape
    print(f"Số lượng mẫu (m): {m}, Số lượng đặc trưng (n): {n}")

    # Thêm cột bias (x0 = 1)
    X_b = np.c_[np.ones((m, 1)), X_poly]
    print(f"Kích thước ma trận X_b: {X_b.shape}")

    # Khởi tạo trọng số ngẫu nhiên nhỏ
    w = np.random.randn(X_b.shape[1], 1) * 0.01
    print(f"Trọng số ban đầu: {w.flatten()}")

    # Gradient Descent
    for iteration in range(n_iterations):
        gradients = (2/m) * X_b.T.dot(X_b.dot(w) - y_train)

        # Kiểm tra nếu gradient có giá trị NaN
        if np.isnan(gradients).any():
            raise ValueError(
                "Gradient chứa giá trị NaN! Hãy kiểm tra lại dữ liệu hoặc learning rate.")

        w -= learning_rate * gradients

    print("✅ Huấn luyện hoàn tất!")
    print(f"Trọng số cuối cùng: {w.flatten()}")

    return w


def training(data):
    st.write("## 📊 Chia dữ liệu (Training - Validation - Testing)")

    # Kiểm tra nếu `target_column` chưa tồn tại
    # if "target_column" not in st.session_state or st.session_state.target_column not in data.columns:
    #     # Mặc định chọn cột đầu tiên
    #     st.session_state.target_column = data.columns[0]

    if "target_column" not in st.session_state:
        st.session_state.target_column = data.columns[0]

    # Cho phép chọn cột mục tiêu nhưng không thay đổi session_state sau khi tạo widget
    selected_label = st.selectbox("Chọn cột dự đoán", data.columns,
                                  index=data.columns.get_loc(st.session_state.target_column))

    X = data.drop(columns=[selected_label], axis=1)
    y = data[selected_label]

    # Chỉ cập nhật session_state khi nút được nhấn
    if st.button("Xác nhận cột cần dự đoán"):
        if st.session_state.target_column != selected_label:
            st.session_state.target_column = selected_label
            st.success(f"Đã chọn cột: **{selected_label}** làm biến mục tiêu")

    # Kiểm tra `st.session_state.data`
    if "data" in st.session_state:
        data = st.session_state.data
    else:
        st.error("Hãy tải tập dữ liệu lên để sử dụng tính năng!")
        st.stop()

    test_size = st.slider("Chọn % testing", 10, 50, 20)
    val_size = st.slider("Chọn % validation", 0, 50, 15)
    remaining_size = 100 - test_size

    if st.button("Chia dữ liệu"):

        stratify_option = y if y.nunique() > 1 else None
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X, y, test_size=test_size/100, stratify=stratify_option, random_state=42
        )

        stratify_option = y_train_full if y_train_full.nunique() > 1 else None
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=val_size / (100 - test_size),
            stratify=stratify_option, random_state=42
        )

        # Lưu vào session_state
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.y = y
        st.session_state.X_val = X_val
        st.session_state.X_train_shape = X_train.shape[0]
        st.session_state.X_val_shape = X_val.shape[0]
        st.session_state.X_test_shape = X_test.shape[0]

    # Chia tập Train - Validation bằng Cross Validation (KFold)
    k = st.slider("Chọn số k trong Cross Validation", 2, 10, 5)
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    st.write("### Tỷ lệ chia dữ liệu")
    if "X_train" in st.session_state and "X_val" in st.session_state and "X_test" in st.session_state:
        table_size = pd.DataFrame({
            'Dataset': ['Train', 'Validation', 'Test'],
            'Kích thước (%)': [remaining_size - val_size, val_size, test_size],
            'Số lượng mẫu': [st.session_state.X_train.shape[0], st.session_state.X_val.shape[0], st.session_state.X_test.shape[0]]
        })
        st.write(table_size)
        st.write(f" - Số fold Cross Validation: {k}")
    else:
        st.warning("⚠️ Vui lòng chia dữ liệu trước khi hiển thị thông tin!")

    st.markdown("---")
    learning_rate = st.slider("Chọn tốc độ học (learning rate):",
                              min_value=1e-6, max_value=0.1, value=0.01, step=1e-6, format="%.6f")
    st.write("## ⚙️ Huấn luyện mô hình")

    # model_option = st.selectbox(
    #     "Chọn mô hình", ["Multiple Regression", "Polynomial Regression"])
    model_type_V = st.selectbox("Chọn loại mô hình:", [
        "Multiple Regression", "Polynomial Regression"])
    model_option = "linear" if model_type_V == "Multiple Regression" else "polynomial"

    degree = 2
    if model_option == "polynomial":
        degree = st.slider("Chọn bậc của Polynomial Regression", 2, 5, 3)

    fold_mse = []
    scaler = StandardScaler()

    if "X_train" not in st.session_state or st.session_state.X_train is None:
        st.warning("⚠️ Vui lòng chia dữ liệu trước khi huấn luyện mô hình!")
        return None, None, None

    X_train, X_test = st.session_state.X_train, st.session_state.X_test
    y_train, y_test = st.session_state.y_train, st.session_state.y_test

    mlflow_input()

    st.session_state["run_name"] = f"{model_option}_run_default"
    if "run_counter" not in st.session_state:
        st.session_state["run_counter"] = 1
    else:
        st.session_state["run_counter"] += 1

    st.session_state["run_name"] = f"{model_option}_run_{st.session_state['run_counter']}"

    if st.button("Huấn luyện mô hình"):

        # if st.button("Huấn luyện mô hình"):
        # 🎯 **Tích hợp MLflow**
        with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}"):
            data = st.session_state.data
            mlflow.log_param("dataset_shape", data.shape)
            mlflow.log_param("target_column", st.session_state.y.name)
            mlflow.log_param("test_size", st.session_state.X_test_shape)
            mlflow.log_param("validation_size",
                             st.session_state.X_val_shape)
            mlflow.log_param("train_size", st.session_state.X_train_shape)

            # Lưu dataset tạm thời
            dataset_path = "dataset.csv"
            data.to_csv(dataset_path, index=False)

            # Log dataset lên MLflow
            mlflow.log_artifact(dataset_path)

            mlflow.log_param("model_option", model_option)
            mlflow.log_param("n_folds", k)
            mlflow.log_param("learning_rate", learning_rate)
            if model_option == "polynomial":
                mlflow.log_param("degree", degree)

            for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train, y_train)):
                X_train_fold, X_valid = X_train.iloc[train_idx], X_train.iloc[valid_idx]
                y_train_fold, y_valid = y_train.iloc[train_idx], y_train.iloc[valid_idx]

                if model_option == "linear":
                    w = train_multiple_linear_regression(
                        X_train_fold, y_train_fold, learning_rate=learning_rate)
                    w = np.array(w).reshape(-1, 1)
                    X_valid_b = np.c_[
                        np.ones((len(X_valid), 1)), X_valid.to_numpy()]
                    y_valid_pred = X_valid_b.dot(w)
                else:
                    X_train_fold = scaler.fit_transform(X_train_fold)
                    w = train_polynomial_regression(
                        X_train_fold, y_train_fold, degree, learning_rate=learning_rate)
                    w = np.array(w).reshape(-1, 1)
                    X_valid_scaled = scaler.transform(X_valid.to_numpy())
                    X_valid_poly = np.hstack(
                        [X_valid_scaled] + [X_valid_scaled**d for d in range(2, degree + 1)])
                    X_valid_b = np.c_[
                        np.ones((len(X_valid_poly), 1)), X_valid_poly]
                    y_valid_pred = X_valid_b.dot(w)

                mse = mean_squared_error(y_valid, y_valid_pred)
                fold_mse.append(mse)
                mlflow.log_metric(f"mse_fold_{fold+1}", mse)
                print(f"📌 Fold {fold + 1} - MSE: {mse:.4f}")

            avg_mse = np.mean(fold_mse)

            if model_option == "linear":
                final_w = train_multiple_linear_regression(
                    X_train, y_train, learning_rate=learning_rate)
                st.session_state['linear_model'] = final_w
                X_test_b = np.c_[
                    np.ones((len(X_test), 1)), X_test.to_numpy()]
                y_test_pred = X_test_b.dot(final_w)
            else:
                X_train_scaled = scaler.fit_transform(X_train)
                final_w = train_polynomial_regression(
                    X_train_scaled, y_train, degree, learning_rate=learning_rate)
                st.session_state['polynomial_model'] = final_w
                X_test_scaled = scaler.transform(X_test.to_numpy())
                X_test_poly = np.hstack(
                    [X_test_scaled] + [X_test_scaled**d for d in range(2, degree + 1)])
                X_test_b = np.c_[
                    np.ones((len(X_test_poly), 1)), X_test_poly]
                y_test_pred = X_test_b.dot(final_w)

            test_mse = mean_squared_error(y_test, y_test_pred)

            # 📌 **Log các giá trị vào MLflow**
            mlflow.log_metric("avg_mse", avg_mse)
            mlflow.log_metric("test_mse", test_mse)

            # Kết thúc run
            mlflow.end_run()

            st.success(f"MSE trung bình qua các folds: {avg_mse:.4f}")
            st.success(f"MSE trên tập test: {test_mse:.4f}")
            st.success(
                f"Log dữ liệu **Train_{st.session_state['run_name']}_{model_option}** thành công!")
            # st.markdown(
            #     f"### 🔗 [Truy cập MLflow mục Linear Regression để xem tham số]({st.session_state['mlflow_url']})")

        return final_w, avg_mse, scaler
    return None, None, None
