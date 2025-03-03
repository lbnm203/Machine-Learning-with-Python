import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def remove_not_required_features(data):
    st.write(
        "### 1️⃣ Loại bỏ cột (đặc trưng) không mang lại hiệu quả trong việc dự đoán")

    st.markdown("""
    Ta sẽ loại bỏ 4 cột (đặc trưng) 
    - **PassengerID**: đây chỉ là số thứ tự hành khách, không ảnh hưởng đến khả năng sống sót.
    - **Cabin**: giá trị thiếu chiếm đến ~80% (687/891)
    - **Ticket**: không mang nhiều thông tin trong quá trình dự đoán
    - **Name**: không cần thiết cho việc dự đoán kết quả
    """)

    if "data" not in st.session_state:
        st.session_state.data = data  # Lưu vào session_state nếu chưa có

    data = st.session_state.data
    columns_to_drop = st.multiselect(
        "Chọn cột muốn xóa:", data.columns.tolist(), key='remove_feature_column_1')

    if st.button("Xóa cột"):
        if columns_to_drop:
            # Tạo bản sao thay vì inplace=True
            data = data.drop(columns=columns_to_drop)
            st.session_state.data = data  # Cập nhật session_state
            st.success(f"✅ Xóa cột {', '.join(columns_to_drop)} thành công")
            st.dataframe(data.head())
        else:
            st.warning("Hãy chọn ít nhất một cột để xóa!")

    return data


def process_missing_values(data):
    st.write("### 2️⃣ Xử lý giá trị thiếu")

    st.markdown("""
        - **Age**: Điền giá trị trung bình cho các giá trị bị thiếu.
        - **Embarked**: Điền giá trị phổ biến nhất (mode) cho các giá trị bị thiếu.
    """)

    if "data" not in st.session_state:
        st.error("❌ Lỗi: Dữ liệu không tồn tại trong session_state!")
        st.session_state.data = data.copy()

    data = st.session_state.data

    # Tìm cột có giá trị thiếu
    missing_cols = data.columns[data.isnull().any()].tolist()
    if not missing_cols:
        st.success("Tập dữ liệu không chứa giá trị thiếu!")
        return data

    selected_col = st.selectbox(
        "Chọn cột cần xử lý:", missing_cols, key="missing_values")
    method = st.radio("Chọn phương pháp xử lý:", [
                      "Mean (trung bình)", "Median (trung vị)", "Mode (giá trị phổ biến nhất)", "Xóa giá trị thiếu"])

    if data[selected_col].dtype == 'object' and method in ["Mean (trung bình)"]:
        st.warning(
            "❗Cột bạn đang chọn chứa dữ liệu dạng chuỗi. Hãy mã hóa thành số thứ tự để xử lý")

    if data[selected_col].dtype == 'object' and method in ["Median (trung vị)"]:
        st.warning(
            "❗Cột bạn đang chọn chứa dữ liệu dạng chuỗi. Hãy mã hóa thành số thứ tự để xử lý")

    if st.button("Xử lý"):
        if data[selected_col].dtype == 'object':
            if method == "Mean (trung bình)":
                unique_values = data[selected_col].dropna().unique()
                encoding_map = {val: idx for idx,
                                val in enumerate(unique_values)}
                data[selected_col] = data[selected_col].map(encoding_map)
                data[selected_col] = data[selected_col].fillna(
                    data[selected_col].mean())

            elif method == "Median (trung vị)":
                unique_values = data[selected_col].dropna().unique()
                encoding_map = {val: idx for idx,
                                val in enumerate(unique_values)}
                data[selected_col] = data[selected_col].map(encoding_map)
                data[selected_col] = data[selected_col].fillna(
                    data[selected_col].median())

            elif method == "Mode (giá trị phổ biến nhất)":
                if not data[selected_col].mode().empty:
                    data[selected_col] = data[selected_col].fillna(
                        data[selected_col].mode()[0])

            elif method == "Xóa giá trị thiếu":
                data = data.dropna(subset=[selected_col])

        else:
            if method == "Mean (trung bình)":
                data[selected_col] = data[selected_col].fillna(
                    data[selected_col].mean())
            elif method == "Median (trung vị)":
                data[selected_col] = data[selected_col].fillna(
                    data[selected_col].median())
            elif method == "Mode (giá trị phổ biến nhất)":
                data[selected_col] = data[selected_col].fillna(
                    data[selected_col].mode()[0])
            elif method == "Xóa giá trị thiếu":
                data = data.dropna(subset=[selected_col])

        st.session_state.data = data
        st.success(f"✅ Xử lý cột `{selected_col}` thành công!")

    st.dataframe(data)
    return data


def convert_data_types(data):
    st.write("### 3️⃣ Chuyển đổi kiểu dữ liệu")

    st.markdown("""
        - **Sex**: Chuyển đổi giá trị của cột Sex từ dạng categorical sang dạng numerical ```(0: male, 1: female)```.
        - **Embarked**: tương tự ở phần này ta sẽ tiến hành chuyển các giá trị về dạng numerical ```(1: S, 2: C, 3: Q)```

    """)

    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()

    if not categorical_cols:
        st.success("✅ Không chứa dữ liệu dạng chuỗi!")
        return data

    selected_col = st.selectbox(
        "Chọn cột muốn chuyển đổi:", categorical_cols, key="encode_data")
    unique_values = data[selected_col].unique()

    # Khởi tạo session_state nếu chưa có
    if "text_inputs" not in st.session_state:
        st.session_state.text_inputs = {}

    if "mapping_dicts" not in st.session_state:
        st.session_state.mapping_dicts = []

    mapping_dict = {}
    input_values = []  # Danh sách để kiểm tra trùng lặp
    # has_duplicate = False  # Biến kiểm tra trùng lặp

    if len(unique_values) < 5:
        for val in unique_values:
            key = f"{selected_col}_{val}"
            if key not in st.session_state.text_inputs:
                st.session_state.text_inputs[key] = ""

            new_val = st.text_input(f"Nhập giá trị thay thế `{val}`:",
                                    key=key,
                                    value=st.session_state.text_inputs[key])

            # Cập nhật session_state với giá trị nhập mới
            st.session_state.text_inputs[key] = new_val
            input_values.append(new_val)

            # Lưu vào mapping_dict nếu không trùng lặp
            mapping_dict[val] = new_val

        # # Kiểm tra nếu có giá trị trùng nhau
        # duplicate_values = [
        #     val for val in input_values if input_values.count(val) > 1 and val != ""]
        # if duplicate_values:
        #     st.warning(
        #         f"Không nhập giá trị trùng nhau! - [{', '.join(set(duplicate_values))}]")

        # # Nút button bị mờ nếu có giá trị trùng lặp
        # btn_disabled = has_duplicate

        if any(val == "" for val in input_values):
            st.warning(
                "⚠️ Hãy nhập giá trị thay thế cho tất cả các giá trị duy nhất trước khi chuyển đổi!")
            # Kiểm tra nếu có giá trị trùng nhau
        duplicate_values = [
            val for val in input_values if input_values.count(val) > 1 and val != ""]
        if duplicate_values:
            st.warning(
                f"Không nhập giá trị trùng nhau! - [{', '.join(set(duplicate_values))}]")

        else:
            if st.button("Chuyển đổi"):
                column_info = {
                    "column_name": selected_col,
                    "mapping_dict": mapping_dict
                }
                st.session_state.mapping_dicts.append(column_info)

                data[selected_col] = data[selected_col].map(
                    lambda x: mapping_dict.get(x, x))
                data[selected_col] = pd.to_numeric(
                    data[selected_col], errors='coerce')

                st.session_state.text_inputs.clear()

                st.session_state.data = data
                st.success(f"✅ Chuyển đổi cột `{selected_col}` thành công")

    st.dataframe(data)
    return data


def scaler_numerical_data(data):
    st.write("### 4️⃣ Chuẩn hóa dữ liệu số")

    st.markdown("""
        Ta sẽ sử dụng phương pháp StandardScaler để chuẩn hóa các giá trị
        khác nhau về cùng khoảng giá trị để tính toán
    """)

    # Lọc tất cả các cột số
    numerical_cols = data.select_dtypes(include=['number']).columns.tolist()

    if not numerical_cols:
        st.success("✅ Không có cột số nào để chuẩn hóa!")
        return data

    # Cho phép người dùng chọn các cột cần chuẩn hóa
    target_col = st.selectbox("Chọn cột mục tiêu:",
                              numerical_cols, key="target_column")

    # Tìm các cột nhị phân (chỉ chứa 0 và 1)
    # binary_cols = [col for col in numerical_cols if data[col].dropna().isin([0, 1]).all()]

    # Loại bỏ cột nhị phân và cột mục tiêu khỏi danh sách cần chuẩn hóa
    cols_to_scale = list(set(numerical_cols) - {target_col})

    # Cho phép người dùng chọn các cột cần chuẩn hóa
    cols_to_scale = st.multiselect(
        "Chọn các cột cần chuẩn hóa:", cols_to_scale, key="scale_columns")

    if not cols_to_scale:
        st.warning("⚠️ Hãy chọn ít nhất một cột để chuẩn hóa!")
        return data

    if st.button("Chuẩn hóa"):
        scaler = StandardScaler()
        data[cols_to_scale] = scaler.fit_transform(data[cols_to_scale])

        # Lưu vào session_state
        st.session_state.data = data

        st.success(
            f"✅ Đã chuẩn hóa các cột : {', '.join(cols_to_scale)}")

        st.dataframe(data)

    return data, target_col


def preprocess_data(data):
    st.markdown("## ⚙️ Quá trình tiền xử lý dữ liệu")

    if "data" not in st.session_state:
        st.session_state.data = data.copy()

    data = remove_not_required_features(data)
    data = process_missing_values(data)
    data = convert_data_types(data)
    data, target_col = scaler_numerical_data(data)
    data = data.copy()
    return data, target_col
