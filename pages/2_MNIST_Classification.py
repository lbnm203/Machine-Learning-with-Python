import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

from streamlit_drawable_canvas import st_canvas
from services.mnist_classfier.utils.theory import theory_info, mnist_dataset
from services.mnist_classfier.utils.training import train_process
from services.mnist_classfier.utils.demo_st import demo_app

# def process_canvas_image(image):
#     # Chuyển đổi ảnh canvas thành grayscale (28x28)
#     image = cv2.resize(image, (28, 28))  # Resize về 28x28
#     image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)  # Chuyển thành grayscale
#     image = image.astype(np.float32) / 255.0  # Chuẩn hóa về [0,1]
#     image = image.reshape(1, -1)  # Chuyển thành vector (1, 784)
#     return image


def process_canvas_image(image):
    """Tiền xử lý ảnh vẽ từ canvas để phù hợp với MNIST"""
    # Chuyển đổi ảnh thành grayscale
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    # Đảo ngược màu (MNIST có chữ trắng trên nền đen)
    image = cv2.bitwise_not(image)

    # Tìm bounding box chứa chữ số
    x, y, w, h = cv2.boundingRect(image)
    if w > 0 and h > 0:
        image = image[y:y+h, x:x+w]  # Cắt vùng chứa chữ số

    # Resize về 28x28 (giữ tỷ lệ chữ số)
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)

    # Làm mịn ảnh để tránh nhiễu
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # Chuẩn hóa về khoảng [0, 1] giống như dữ liệu MNIST
    image = image.astype(np.float32) / 255.0

    # Chuyển thành vector (1, 784) để đưa vào mô hình
    image = image.reshape(1, -1)

    return image


# Huấn luyện và đánh giá mô hình


def train_and_evaluate(model_name, X_train, X_test, y_train, y_test):
    if model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "SVM":
        model = SVC(gamma='scale', random_state=42)
    else:
        st.error("Chọn mô hình hợp lệ!")
        return None, None

    # Bắt đầu MLFlow
    with mlflow.start_run():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        # Log vào MLFlow
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

        st.write(f"Accuracy: {accuracy:.4f}")

    return model, accuracy


# Streamlit UI
def main():
    st.title("Ứng dụng phân lớp MNIST với Streamlit & MLFlow")

    data_mnist, theory, train, demo = st.tabs(
        ["Tập dữ liệu", "Thông tin", "Huấn Luyện", "Demo"])

    # --------------- Data MNIST ---------------
    with data_mnist:
        X, y = mnist_dataset()

    # -------- Theory Decision Tree - SVM ---------
    with theory:
        pass

    # --------------- Training ---------------
    with train:
        train_process(X, y)

    # --------------- DEMO MNIST ---------------
    with demo:
        demo_app()


if __name__ == "__main__":
    main()
