import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from services.Linear_Regression.utils.preprocess import preprocess_data


def demo_app(data):
    model_type = st.selectbox(
        "Chọn mô hình:", ["Multiple Regression", "Polynomial Regression"])

    if model_type == "Multiple Regression" and "linear_model" in st.session_state:
        model = st.session_state["linear_model"]
    elif model_type == "Polynomial Regression" and "polynomial_model" in st.session_state:
        model = st.session_state["polynomial_model"]
    else:
        st.warning("Mô hình chưa được huấn luyện.")
        return

    # Nhập các giá trị cho các cột của X_train
    X_train = st.session_state.X_train

    st.write(X_train.head())

    # Đảm bảo bạn dùng session_state
    num_columns = len(X_train.columns)
    column_names = X_train.columns.tolist()

    st.write(f"Nhập các giá trị cho {num_columns} cột của X_train:")

    # Tạo các trường nhập liệu cho từng cột
    X_train_input = []
    binary_columns = []

    if "mapping_dicts" not in st.session_state:
        st.session_state.mapping_dicts = []

    # Chia thành 2 cột
    cols = st.columns(2)

    # Duyệt qua 8 cột đầu tiên (giới hạn hiển thị 4 dòng x 2 cột)
    # Chỉ lấy 8 cột đầu tiên
    for i, column_name in enumerate(column_names[:8]):
        mapping_dict = None
        for column_info in st.session_state.mapping_dicts:
            if column_info["column_name"] == column_name:
                mapping_dict = column_info["mapping_dict"]
                break

        # Hiển thị trong cột trái hoặc phải
        with cols[i % 2]:  # Cột 1 hoặc cột 2 luân phiên
            if mapping_dict:  # Nếu có mapping_dict, hiển thị dropdown
                value = st.selectbox(
                    f"Giá trị cột {column_name}",
                    options=list(mapping_dict.keys()),
                    key=f"column_{i}"
                )
                value = int(mapping_dict[value])
            else:  # Nếu không có mapping_dict, hiển thị ô nhập số
                value = st.number_input(
                    f"Giá trị cột {column_name}", key=f"column_{i}")

        X_train_input.append(value)

    # Chuyển đổi list thành array
    X_train_input = np.array(X_train_input).reshape(1, -1)

    # Sao chép X_train_input để thay đổi giá trị không làm ảnh hưởng đến dữ liệu gốc
    X_train_input_final = X_train_input.copy()
    scaler = StandardScaler()
    # Tạo mảng chỉ số của các phần tử khác 0 và 1
    for i in range(X_train_input.shape[1]):

        # Nếu giá trị không phải 0 hoặc 1
        if X_train_input[0, i] != 0 and X_train_input[0, i] != 1:
            # Chuẩn hóa giá trị

            X_train_input_final[0, i] = scaler.fit_transform(
                X_train_input[:, i].reshape(-1, 1)).flatten()

    st.write("Dữ liệu sau khi xử lý:")

    if st.button("Dự đoán"):
        # Thêm cột 1 cho intercept (nếu cần)
        X_input_b = np.c_[
            np.ones((X_train_input_final.shape[0], 1)), X_train_input_final]

        # Dự đoán với mô hình đã lưu

        y_pred = X_input_b.dot(model)  # Dự đoán với mô hình đã lưu

        # Hiển thị kết quả dự đoán
        if y_pred >= 0.5:
            st.write("#### Sống sót 🥹")
        else:
            st.write("#### Không Sống Sót 💀")
