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
                      "Mean (trung bình)", "Median (trung vị)", "Mode (giá trị phổ biến nhất)"])

    if st.button("Xử lý"):
        if method == "Mean (trung bình)":
            data[selected_col].fillna(
                data[selected_col].mean(), inplace=True)
        elif method == "Median (trung vị)":
            data[selected_col].fillna(
                data[selected_col].median(), inplace=True)
        elif method == "Mode (giá trị phổ biến nhất)":
            if not data[selected_col].mode().empty:
                data[selected_col].fillna(
                    data[selected_col].mode()[0], inplace=True)
            else:
                st.warning(f"⚠️ Không tìm thấy mode cho cột `{selected_col}`!")

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
    unique_values = data[selected_col].dropna().unique()

    mapping_dict = {}
    if len(unique_values) < 10:
        for val in unique_values:
            new_val = st.text_input(
                f"Thay thế `{val}`:", key=f"{selected_col}_{val}")
            if new_val.strip():
                mapping_dict[val] = new_val

        if st.button("Chuyển đổi"):

            data[selected_col] = data[selected_col].map(
                lambda x: mapping_dict.get(x, x))

            try:
                data[selected_col] = pd.to_numeric(
                    data[selected_col], errors='coerce')
            except Exception as e:
                st.error(f"❌ Lỗi khi chuyển đổi: {e}")

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
    cols_to_scale = st.multiselect(
        "Chọn cột cần chuẩn hóa:", numerical_cols, default=numerical_cols, key="scaler_data"
    )

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

    return data


def preprocess_data(data):
    st.markdown("## ⚙️ Quá trình tiền xử lý dữ liệu")

    if "data" not in st.session_state:
        st.session_state.data = data.copy()

    data = remove_not_required_features(data)
    data = process_missing_values(data)
    data = convert_data_types(data)
    data = scaler_numerical_data(data)
    data = data.copy()
    return data


# def preprocess_data(df):
#     """Tiền xử lý dữ liệu Titanic."""

#     df = df.copy()

#     # 1. Xử lý missing values
#     df['Age'].fillna(df['Age'].median(), inplace=True)
#     df['Fare'].fillna(df['Fare'].median(), inplace=True)
#     df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

#     # Drop 3 đặc trưng Ticket, Cabin, Name
#     df.drop(columns=['Ticket', 'Cabin', 'Name'], inplace=True)

#     # 4. Encode cho các biến phân loại
#     df['Embarked'] = df['Embarked'].map(
#         {'S': 1, 'C': 2, 'Q': 3}).astype('Int64')
#     df["Sex"] = df["Sex"].map({'male': 0, 'female': 1}).astype('Int64')

#     # 3. Chuẩn hóa các cột số
#     scaler = StandardScaler()
#     numerical_features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Sex', 'Embarked', 'PassengerId', 'Survived']
#     df[numerical_features] = scaler.fit_transform(df[numerical_features])


#     # features = [col for col in df.columns if col != 'Survived']
#     # target = 'Survived'
#     # X = df[features]
#     # y = df[target]
#     X = df.drop(columns=['Survived'])
#     y = df['Survived']

#     # Định nghĩa preprocessor
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), [
#                 'Age', 'Fare', 'Pclass', 'SibSp', 'Parch']),
#             ('cat', OneHotEncoder(), ['Embarked', 'Sex'])
#         ]
#     )

#     return X, y, preprocessor, df
