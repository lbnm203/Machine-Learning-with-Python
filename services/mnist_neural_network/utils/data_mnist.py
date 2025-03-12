import numpy as np
from sklearn.datasets import fetch_openml
import streamlit as st
import matplotlib.pyplot as plt
import pickle


def visualize_mnist(X, y):
    unique_labels = np.unique(y)
    images = []

    # Lấy một ảnh cho mỗi nhãn từ 0 đến 9
    for label in unique_labels:
        idx = np.nonzero(y == label)[0][0]  # Lấy index đầu tiên của label
        images.append((X[idx], label))

    fig, axes = plt.subplots(2, 5, figsize=(7, 3))
    # fig.suptitle("Những nhãn trong tập dữ liệu MNIST", fontsize=10)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i][0].reshape(28, 28), cmap="gray")
        ax.set_title(f"Label: {images[i][1]}")
        ax.axis("off")

    st.pyplot(fig)


# @st.cache_data
@st.cache_data
def mnist_dataset():

    st.markdown("## 📜 Tập dữ liệu MNIST")
    st.write("---")

    with open('./services/mnist_clustering/data/X.pkl', 'rb') as f:
        X = pickle.load(f)

    with open('./services/mnist_clustering/data/y.pkl', 'rb') as f:
        y = pickle.load(f)

    col1, col2 = st.columns(2)
    with col1:
        visualize_mnist(X, y)
        st.write(
            f"Tập dữ liệu MNIST gồm {X.shape[0]} mẫu, {X.shape[1]} đặc trưng")

    with col2:
        st.markdown("""
            **MNIST (Modified National Institute of Standards and Technology)** là một trong 
            những tập dữ liệu phổ biến nhất trong lĩnh vực nhận dạng chữ số viết tay. Đây 
            là một tập dữ liệu tiêu chuẩn để huấn luyện và đánh giá các mô hình machine 
            learning (ML) và deep learning (DL), đặc biệt là các mô hình nhận dạng hình ảnh.
        
            MNIST gồm 70.000 ảnh chữ số viết tay với kích thước ảnh 28x28 pixel, ảnh grayscale(đen trắng, 1 kênh màu)
            - 60.000 ảnh dùng để huấn luyện (training set)
            - 10.000 ảnh dùng để đánh giá (test set)
                         
            - Số lớp (số nhãn): 10 (các chữ số từ 0 đến 9)

            Mỗi ảnh được biểu diễn dưới dạng ma trận 28x28

            Ta sẽ chuẩn hóa dữ liệu, đưa giá trị pixel ban đầu nằm trong khoảng [0, 255], 
            cần chia cho 255.0 để đưa về khoảng [0,1]                        
        """)

    # Visualize target distribution
    fig, ax = plt.subplots(figsize=(7, 3))
    unique, counts = np.unique(y, return_counts=True)
    ax.bar(unique, counts, tick_label=unique)
    ax.set_title("Phân phối các nhãn trong tập dữ liệu MNIST")
    ax.set_xlabel("Nhãn")
    ax.set_ylabel("Số lượng")
    st.pyplot(fig)

    st.write("---")

    return X, y
