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
from sklearn.model_selection import cross_val_score


def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/lbnm203/Machine_Learning_UI.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "lbnm203"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "0902d781e6c2b4adcd3cbf60e0f288a8085c5aab"

    mlflow.set_experiment("MNIST_Classification")


def train_process(X, y):
    mlflow_input()
    st.write("## ⚙️ Quá trình huấn luyện")

    total_samples = X.shape[0]

    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("Chọn số lượng ảnh để train:",
                            1000, total_samples, 10000)

    st.session_state.total_samples = num_samples
    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("Chọn % dữ liệu Test", 10, 50, 20)
    val_size = st.slider("Chọn % dữ liệu Validation", 0, 50,
                         10)  # Xác định trước khi sử dụng

    remaining_size = 100 - test_size  # Sửa lỗi: Sử dụng test_size thay vì train_size

    # Chọn số lượng ảnh theo yêu cầu
    X_selected, _, y_selected, _ = train_test_split(
        X, y, train_size=num_samples, stratify=y, random_state=42
    )

    # Chia train/test theo tỷ lệ đã chọn
    stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_selected, y_selected, test_size=test_size/100, stratify=stratify_option, random_state=42
    )

    # Chia train/val theo tỷ lệ đã chọn
    stratify_option = y_train_full if len(
        np.unique(y_train_full)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size / remaining_size,
        stratify=stratify_option, random_state=42
    )

    # Lưu vào session_state
    st.session_state.X_train = X_train
    st.session_state.X_val = X_val
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_val = y_val
    st.session_state.y_test = y_test
    st.session_state.test_size = X_test.shape[0]
    st.session_state.val_size = X_val.shape[0]
    st.session_state.train_size = X_train.shape[0]

    if "X_train" in st.session_state:
        X_train = st.session_state.X_train
        X_val = st.session_state.X_val
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_val = st.session_state.y_val
        y_test = st.session_state.y_test

    # st.write(f'Training: {X_train.shape[0]} - Testing: {y_test.shape[0]}')

    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    table_size = pd.DataFrame({
        'Dataset': ['Train', 'Validation', 'Test'],
        'Kích thước (%)': [remaining_size - val_size, val_size, test_size],
        'Số lượng mẫu': [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
    })
    st.write(table_size)

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
        st.markdown("""- criterion: xác định cách thức cây quyết định chọn thuộc tính để phân nhánh""", help="""
- Gini:
    - Mục tiêu của Gini là chọn thuộc tính phân nhánh sao cho dữ liệu sau khi chia có độ thuần khiết cao nhất.
    - Nếu một node chỉ chứa mẫu của một lớp duy nhất, giá trị Gini sẽ là 0 (thuần khiết hoàn toàn).
    - Nếu một node chứa mẫu của nhiều lớp khác nhau, giá trị Gini sẽ tăng.
- Entropy:
    - Mục tiêu của entropy là chọn thuộc tính phân nhánh sao cho độ không chắc chắn của dữ liệu giảm nhiều nhất.
    - Entropy càng cao ⟶ Dữ liệu càng hỗn loạn (ít thuần khiết).
    - Entropy càng thấp ⟶ Dữ liệu càng thuần khiết.
 """)
        st.markdown("- min_samples_leaf (Số lượng mẫu tối thiểu trong mỗi lá) ", help="""
- Mục tiêu là quy định số lượng mẫu nhỏ nhất mà một node lá có thể có.
- Nếu một node có ít hơn min_samples_leaf mẫu, nó sẽ bị hợp nhất với node cha thay vì trở thành một node lá.
- Giúp tránh overfitting bằng cách ngăn cây quyết định quá phức tạp.
""")

        max_depth = st.slider("max_depth", 1, 20, 5)
        criterion = st.selectbox("Chọn tiêu chí phân nhánh (criterion)", [
            "gini", "entropy"], index=0)
        min_samples_leaf = st.slider(
            "Số lượng mẫu tối thiểu trên mỗi lá (min_samples_leaf)", 1, 10, 2)
        model = DecisionTreeClassifier(
            max_depth=max_depth, criterion=criterion, min_samples_leaf=min_samples_leaf)

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

    n_folds = st.slider("Chọn số folds Cross-Validation:",
                        min_value=2, max_value=10, value=3)

    if st.button("Huấn luyện mô hình"):
        with mlflow.start_run(run_name=f"Train_{st.session_state['run_name']}"):
            mlflow.log_param("test_size", st.session_state.test_size)
            mlflow.log_param("val_size", st.session_state.val_size)
            mlflow.log_param("train_size", st.session_state.train_size)
            mlflow.log_param("num_samples", st.session_state.total_samples)

            cv_scores = cross_val_score(model, X_train, y_train, cv=n_folds)
            mean_cv_score = cv_scores.mean()
            std_cv_score = cv_scores.std()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            st.success(f"Độ chính xác trên Testing: {acc:.4f}")
            st.success(
                f"Độ chính xác trung bình khi Cross-Validation: {mean_cv_score:.4f}                                                     ")

            mlflow.log_param("model", model_choice)
            if model_choice == "Decision Tree":
                mlflow.log_param("max_depth", max_depth)
            elif model_choice == "SVM":
                mlflow.log_param("C", C)
                mlflow.log_param("kernel", kernel)

            mlflow.log_metric("test_accuracy", acc)
            mlflow.log_metric("cv_accuracy_mean", mean_cv_score)
            mlflow.log_metric("cv_accuracy_std", std_cv_score)
            mlflow.sklearn.log_model(model, model_choice.lower())

            st.success("Lưu tham số vào MLflow thành công!")

        if model_choice == "Decision Tree":
            depths = range(1, 21)
            accuracies = []
            for depth in depths:
                model = DecisionTreeClassifier(
                    max_depth=max_depth, criterion=criterion, min_samples_leaf=min_samples_leaf)
                model.fit(X_train, y_train)
                y_temp_pred = model.predict(X_test)
                temp_acc = accuracy_score(y_test, y_temp_pred)
                accuracies.append(temp_acc)

            mlflow.log_param("criterion", criterion)
            mlflow.log_param("min_samples_leaf", min_samples_leaf)

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
        st.success(
            f"Log dữ liệu **{st.session_state['models']}** thành công!")
        # st.write(f"🔹 Mô hình đã được lưu với tên: {model_name}")
        # st.write(
        #     f"Tổng số mô hình hiện tại: {len(st.session_state['models'])}")

        # # In tên các mô hình đã lưu
        # st.write("📋 Danh sách các mô hình đã lưu:")
        model_names = [model["name"] for model in st.session_state["models"]]
        # # Hiển thị tên các mô hình trong một d
        # st.write(", ".join(model_names))

        # st.success("Lưu thành công!")

        # st.markdown(
        #     f"🔗 [Truy cập MLflow UI MNIST_Classification để xem tham số]({st.session_state['mlflow_url']})")
