import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os

# 🌟 Kết nối với DagsHub MLflow
DAGSHUB_MLFLOW_URI = "https://dagshub.com/lbnm203/Machine_Learning_UI.mlflow"
st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)

os.environ["MLFLOW_TRACKING_USERNAME"] = "lbnm203"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "0902d781e6c2b4adcd3cbf60e0f288a8085c5aab"

# 📝 Kiểm tra danh sách các experiment có sẵn
client = MlflowClient()
experiments = client.search_experiments()
experiment_name = "MNIST_Classification"

if not any(exp.name == experiment_name for exp in experiments):
    mlflow.create_experiment(experiment_name)
    st.success(f"Experiment '{experiment_name}' đã được tạo!")
else:
    st.info(f"Experiment '{experiment_name}' đã tồn tại.")

mlflow.set_experiment(experiment_name)


def train_process(X, y):
    st.write("## ⚙️ Quá trình huấn luyện")

    total_samples = X.shape[0]

    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("Chọn số lượng ảnh để train:",
                            1000, total_samples, 10000)

    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("Chọn tỷ lệ test:", 0.1, 0.5, 0.2)

    X_selected, y_selected = X[:num_samples], y[:num_samples]

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_selected, test_size=test_size, random_state=42)

    # Lưu vào session_state để sử dụng sau
    st.session_state["X_train"] = X_train
    st.session_state["y_train"] = y_train
    st.session_state["X_test"] = X_test
    st.session_state["y_test"] = y_test

    if "X_train" in st.session_state:
        X_train = st.session_state["X_train"]
        # st.write(X_train.dtype)
        y_train = st.session_state["y_train"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]
    else:
        st.error("⚠️ Chưa có dữ liệu! Hãy chia dữ liệu trước.")
        return

    st.write(f'Training: {X_train.shape[0]} - Testing: {y_test.shape[0]}')

    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    st.write("---")

    # 📌 **Chọn mô hình**
    model_choice = st.selectbox("Chọn mô hình:", ["Decision Tree", "SVM"])

    if model_choice == "Decision Tree":
        st.markdown("""
        - **Tham số mô hình:**  
            - **max_depth**: Độ sâu tối đa của cây.  
                - **Giá trị nhỏ**: Tránh overfitting nhưng có thể underfitting.  
                - **Giá trị lớn**: Dễ bị overfitting vì khó học được các mẫu phức tạp trong dữ liệu 
        """)

        max_depth = st.slider("max_depth", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth)

    elif model_choice == "SVM":
        st.markdown("""
        - **Tham số mô hình:**  
            - **C (Regularization)**: Hệ số điều chỉnh độ phạt lỗi.  
                - **C nhỏ**: Mô hình đơn giản hơn, chấp nhận nhiều lỗi hơn.  
                - **C lớn**: Mô hình cố gắng phân loại chính xác mọi điểm, nhưng dễ bị overfitting.  
            - **Kernel**: Hàm kernel trick giúp phân tách dữ liệu phi tuyến tính bằng cách ánh xạ dữ 
                liệu vào không gian có nhiều chiều hơn.
                - Linear: Mô hình dùng siêu phẳng tuyến tính để phân lớp.  
                - RBF: Kernel Gaussian Radial Basis Function giúp phân tách dữ liệu phi tuyến tính tốt hơn.  
                - Poly: Sử dụng đa thức bậc cao để phân lớp, phù hợp với dữ liệu có mối quan hệ phức tạp.  
                - Sigmoid: Mô phỏng hàm kích hoạt của mạng nơ-ron.
 
        """)
        C = st.slider("C (Regularization)", 0.1, 10.0, 1.0)
        kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"])
        model = SVC(C=C, kernel=kernel)

    if st.button("Huấn luyện mô hình"):
        with mlflow.start_run():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Độ chính xác trên Testing: {acc:.4f}")

            mlflow.log_param("model", model_choice)
            if model_choice == "Decision Tree":
                mlflow.log_param("max_depth", max_depth)
            elif model_choice == "SVM":
                mlflow.log_param("C", C)
                mlflow.log_param("kernel", kernel)

            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(model, model_choice.lower())

            st.success("Lưu tham số vào MLflow thành công!")

        if model_choice == "Decision Tree":
            depths = range(1, 21)
            accuracies = []
            for depth in depths:
                model = DecisionTreeClassifier(max_depth=max_depth)
                model.fit(X_train, y_train)
                y_temp_pred = model.predict(X_test)
                temp_acc = accuracy_score(y_test, y_temp_pred)
                accuracies.append(temp_acc)

            # st.write("Độ chính xác qua từng độ sâu ")
            # accuracy_df = pd.DataFrame(
            #     {"Độ sâu": depths, "Độ chính xác": accuracies})
            # st.line_chart(accuracy_df.set_index("Độ sâu"))

        # st.success(f"✅ Độ chính xác: {temp_acc:.4f}")

        # model.fit(X_train, y_train)
        # y_pred = model.predict(X_test)
        # acc = accuracy_score(y_test, y_pred)
        # st.success(f"✅ Độ chính xác: {acc:.4f}")

        # Lưu mô hình vào session_state dưới dạng danh sách nếu chưa có
        if "models" not in st.session_state:
            st.session_state["models"] = []

        # Tạo tên mô hình dựa trên lựa chọn mô hình và kernel
        model_name = model_choice.lower().replace(" ", "_")
        if model_choice == "SVM":
            model_name += f"_{kernel}"

        # Kiểm tra nếu tên mô hình đã tồn tại trong session_state
        existing_model = next(
            (item for item in st.session_state["models"] if item["name"] == model_name), None)

        if existing_model:
            # Tạo tên mới với số đếm phía sau
            count = 1
            new_model_name = f"{model_name}_{count}"

            # Kiểm tra tên mới chưa tồn tại
            while any(item["name"] == new_model_name for item in st.session_state["models"]):
                count += 1
                new_model_name = f"{model_name}_{count}"

            # Sử dụng tên mới đã tạo
            model_name = new_model_name
            # st.warning(f"⚠️ Mô hình được lưu với tên là: {model_name}")

        # # Lưu mô hình vào danh sách với tên mô hình cụ thể
        st.session_state["models"].append({"name": model_name, "model": model})
        # st.write(f"🔹 Mô hình đã được lưu với tên: {model_name}")
        # st.write(
        #     f"Tổng số mô hình hiện tại: {len(st.session_state['models'])}")

        # # In tên các mô hình đã lưu
        # st.write("📋 Danh sách các mô hình đã lưu:")
        model_names = [model["name"] for model in st.session_state["models"]]
        # # Hiển thị tên các mô hình trong một d
        # st.write(", ".join(model_names))

        # st.success("Lưu thành công!")

        st.markdown(
            f"🔗 [Truy cập MLflow UI MNIST_Classification để xem tham số]({st.session_state['mlflow_url']})")
