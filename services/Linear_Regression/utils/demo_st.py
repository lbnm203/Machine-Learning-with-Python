import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from services.Linear_Regression.utils.preprocess import preprocess_data


def demo_app(data):
    # Khởi tạo scaler, polynomial transformer, và model nếu chưa có
    feature_columns = [
        col for col in data.columns if col != "Survived"]

    if "scaler" not in st.session_state:
        st.session_state.scaler = StandardScaler()
        st.session_state.scaler.fit(data[feature_columns])

    if "poly" not in st.session_state:
        st.session_state.poly = PolynomialFeatures(degree=2)
        st.session_state.poly.fit(data[feature_columns])

    if "model_option" not in st.session_state:
        st.session_state.model_option = "Multiple Regression"  # Giá trị mặc định

    if "model" not in st.session_state:
        st.error(
            "⚠️ Mô hình chưa được huấn luyện! Hãy chạy huấn luyện trước khi dự đoán.")

    # 5️⃣ Dự đoán trên dữ liệu mới
    st.markdown("## 🔮 Dự đoán trên dữ liệu mới")
    st.write("Nhập dữ liệu của hành khách mới để dự đoán khả năng sống sót")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            pclass = st.selectbox("Pclass", [1, 2, 3])
            age = st.number_input(
                "Age", min_value=0, max_value=100, value=25)
            fare = st.number_input("Fare", min_value=0, value=100)
        with col2:
            sex = st.selectbox("Sex", ["male", "female"])
            embarked = st.selectbox("Embarked", ["C", "Q", "S"])
            sibsp = st.number_input("Sibsp", min_value=0, value=5)
            parch = st.number_input("Parch", min_value=0, value=5)

        if st.form_submit_button("📊 Dự đoán"):
            try:
                # Chuyển đổi dữ liệu đầu vào thành DataFrame với đúng thứ tự cột
                input_data = pd.DataFrame([[pclass, age, fare, 1 if sex == "female" else 0,
                                            1 if embarked == "C" else 2 if embarked == "Q" else 3,
                                            sibsp, parch]],
                                          columns=feature_columns)

                # Chuẩn hóa dữ liệu
                input_scaled = st.session_state.scaler.transform(
                    input_data)

                if "model" in st.session_state:
                    if st.session_state.model_option == "Multiple Regression":
                        prediction = st.session_state.model.predict(
                            input_scaled)
                    else:
                        input_poly = st.session_state.poly.transform(
                            input_scaled)

                        # Kiểm tra số đặc trưng có khớp không
                        expected_features = st.session_state.poly.n_output_features_
                        if input_poly.shape[1] != expected_features:
                            st.error(
                                f"❌ Số lượng đặc trưng đầu vào ({input_poly.shape[1]}) không khớp với số lượng khi huấn luyện ({expected_features}). Hãy kiểm tra lại quá trình xử lý dữ liệu.")
                        else:
                            prediction = st.session_state.model.predict(
                                input_poly)
                            result = "Sống sót 🟢" if prediction[0] > 0.5 else "Không sống sót 🔴"
                            st.success(f"🔮 Kết quả dự đoán: {result}")
                            st.write(
                                f"🔹 Giá trị dự đoán: {prediction[0]:.4f}")
                else:
                    st.error(
                        "❌ Mô hình chưa được tải! Hãy huấn luyện trước khi dự đoán.")
            except Exception as e:
                st.error(f"❌ Lỗi khi dự đoán: {e}")

    # # Khởi tạo scaler, polynomial transformer, và model nếu chưa có
    # feature_columns = [
    #     col for col in data.columns if col != "Survived"]

    # if "scaler" not in st.session_state:
    #     st.session_state.scaler = StandardScaler()
    #     st.session_state.scaler.fit(preprocess_data[feature_columns])

    # if "poly" not in st.session_state:
    #     st.session_state.poly = PolynomialFeatures(degree=2)
    #     st.session_state.poly.fit(preprocess_data[feature_columns])

    # if "model_option" not in st.session_state:
    #     st.session_state.model_option = "Multiple Regression"  # Giá trị mặc định

    # if "model" not in st.session_state:
    #     st.error(
    #         "⚠️ Mô hình chưa được huấn luyện! Hãy chạy huấn luyện trước khi dự đoán.")

    # # 5️⃣ Dự đoán trên dữ liệu mới
    # st.markdown("## 🔮 Dự đoán trên dữ liệu mới")
    # st.write("Nhập dữ liệu của hành khách mới để dự đoán khả năng sống sót")

    # with st.form("prediction_form"):
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         pclass = st.selectbox("Pclass", [1, 2, 3])
    #         age = st.number_input("Age", min_value=0, max_value=100, value=25)
    #         fare = st.number_input("Fare", min_value=0, value=120)
    #     with col2:
    #         sex = st.selectbox("Sex", ["male", "female"])
    #         embarked = st.selectbox("Embarked", ["C", "Q", "S"])
    #         sibsp = st.number_input("Sibsp", min_value=0, value=5)
    #         parch = st.number_input("Parch", min_value=0, value=5)

    #     if st.form_submit_button("📊 Dự đoán"):
    #         try:
    #             # Chuyển đổi dữ liệu đầu vào thành DataFrame với đúng thứ tự cột
    #             input_data = pd.DataFrame([[pclass, age, fare, 1 if sex == "female" else 0,
    #                                         1 if embarked == "C" else 2 if embarked == "Q" else 3,
    #                                         sibsp, parch]],
    #                                       columns=feature_columns)

    #             # Chuẩn hóa dữ liệu
    #             input_scaled = st.session_state.scaler.transform(input_data)

    #             if "model" in st.session_state:
    #                 if st.session_state.model_option == "Multiple Regression":
    #                     prediction = st.session_state.model.predict(
    #                         input_scaled)
    #                 else:
    #                     input_poly = st.session_state.poly.transform(
    #                         input_scaled)
    #                     prediction = st.session_state.model.predict(input_poly)

    #                 # Kiểm tra số đặc trưng có khớp không
    #                 if input_poly.shape[1] != st.session_state.poly.n_output_features_:
    #                     st.error(
    #                         f"❌ Số lượng đặc trưng đầu vào ({input_poly.shape[1]}) không khớp với số lượng khi huấn luyện ({st.session_state.poly.n_output_features_}). Hãy kiểm tra lại quá trình xử lý dữ liệu.")
    #                 else:
    #                     result = "Sống sót 🟢" if prediction[0] > 0.5 else "Không sống sót 🔴"
    #                     st.success(f"🔮 Kết quả dự đoán: {result}")
    #                     st.write(f"🔹 Giá trị dự đoán: {prediction[0]:.4f}")
    #             else:
    #                 st.error(
    #                     "❌ Mô hình chưa được tải! Hãy huấn luyện trước khi dự đoán.")
    #         except Exception as e:
    #             st.error(f"❌ Lỗi khi dự đoán: {e}")

    # model_option = st.selectbox(
    #     "Chọn mô hình", ["Multiple Regression", "Polynomial Regression"])

    # # Khởi tạo scaler và polynomial transformer nếu chưa có
    # feature_columns = [
    #     col for col in data.columns if col != "Survived"]

    # if "scaler" not in st.session_state:
    #     st.session_state.scaler = StandardScaler()
    #     st.session_state.scaler.fit(preprocess_data[feature_columns])

    # if "poly" not in st.session_state:
    #     st.session_state.poly = PolynomialFeatures(degree=2)
    #     st.session_state.poly.fit(preprocess_data[feature_columns])

    # # 5️⃣ Dự đoán trên dữ liệu mới
    # st.markdown("## 🔮 Dự đoán trên dữ liệu mới")
    # st.write("Nhập dữ liệu của hành khách mới để dự đoán khả năng sống sót")

    # with st.form("prediction_form"):
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         pclass = st.selectbox("Pclass", [1, 2, 3])
    #         age = st.number_input("Age", min_value=0, max_value=100, value=25)
    #         fare = st.number_input("Fare", min_value=0, value=120)
    #     with col2:
    #         sex = st.selectbox("Sex", ["male", "female"])
    #         embarked = st.selectbox("Embarked", ["C", "Q", "S"])
    #         sibsp = st.number_input("Sibsp", min_value=0, value=5)
    #         parch = st.number_input("Parch", min_value=0, value=5)

    #     if st.form_submit_button("📊 Dự đoán"):
    #         try:
    #             # Chuyển đổi dữ liệu đầu vào thành DataFrame với đúng thứ tự cột
    #             input_data = pd.DataFrame([[pclass, age, fare, 1 if sex == "female" else 0,
    #                                         1 if embarked == "C" else 2 if embarked == "Q" else 3,
    #                                         sibsp, parch]],
    #                                       columns=feature_columns)

    #             # Chuẩn hóa dữ liệu
    #             input_scaled = st.session_state.scaler.transform(input_data)

    #             if st.session_state.model_option == "Multiple Regression":
    #                 prediction = st.session_state.model.predict(input_scaled)
    #             else:
    #                 input_poly = st.session_state.poly.transform(input_scaled)
    #                 prediction = st.session_state.model.predict(input_poly)

    #             result = "Sống sót 🟢" if prediction[0] > 0.5 else "Không sống sót 🔴"
    #             st.success(f"🔮 Kết quả dự đoán: {result}")
    #             st.write(f"🔹 Giá trị dự đoán: {prediction[0]:.4f}")
    #         except Exception as e:
    #             st.error(f"❌ Lỗi khi dự đoán: {e}")
