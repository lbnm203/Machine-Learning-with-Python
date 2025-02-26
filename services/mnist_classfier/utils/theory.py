import numpy as np
from sklearn.datasets import fetch_openml
import streamlit as st
import matplotlib.pyplot as plt


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

    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X, y = mnist["data"], mnist["target"]

    # chuyển đổi kiểu dữ liệu
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    col1, col2 = st.columns(2)
    with col1:
        visualize_mnist(X, y)

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


def decision_tree_theory():
    # Tiêu đề chính
    st.header("Thuật toán Decision Tree (Cây Quyết Định)")
    st.write("""
    - **Decision Tree** là một thuật toán học máy dùng cho cả phân loại và hồi quy. Nó chia dữ liệu thành các vùng dựa trên các điều kiện quyết định, được biểu diễn dưới dạng cấu trúc cây với nút gốc, nút nội bộ, và nút lá.

    - **Ứng dụng**: Phân loại email, dự đoán giá nhà, chẩn đoán y khoa.
    """)

    col1, col2 = st.columns(2)
    with col1:

        # Phần 1: Các khái niệm cơ bản
        st.subheader("Các khái niệm cơ bản")
        st.write("""
        - **Đặc trưng (Feature)**: Các biến đầu vào (ví dụ: tuổi, thu nhập).
        - **Nhãn (Label)**: Biến đầu ra cần dự đoán (ví dụ: Có/Không).
        - **Độ không thuần khiết (Impurity)**: Mức độ hỗn tạp của dữ liệu trong một nút.
        - **Chia nhánh (Splitting)**: Quá trình chia dữ liệu dựa trên một điều kiện.
        """)

        # Phần 2: Cách hoạt động
        st.subheader("Cách hoạt động của Decision Tree")
        st.write("""
        1. **Chọn đặc trưng và ngưỡng tốt nhất**:
            - Duyệt qua các đặc trưng và giá trị để tìm cách chia giảm độ không thuần khiết nhiều nhất.
        2. **Chia dữ liệu**:
            - Dữ liệu được chia thành các nhánh dựa trên điều kiện (ví dụ: "tuổi ≤ 30").
        3. **Lặp lại**:
            - Tiếp tục chia trên mỗi nhánh cho đến khi đạt điều kiện dừng (thuần khiết, độ sâu tối đa, số mẫu tối thiểu).
        4. **Gán giá trị nút lá**:
            - Phân loại: Chọn nhãn phổ biến nhất.
            - Hồi quy: Chọn giá trị trung bình.
        """)

    with col2:

        # Phần 3: Công thức toán học
        st.subheader("Công thức toán học")

        st.markdown("""
        #### Gini Index (Phân loại)
        - **Đo độ không thuần khiết:** """)

        st.latex(r"\text{Gini} = 1 - \sum_{i=1}^{c} p_i^2")
        st.write("""
        - $p_i$: Tỷ lệ mẫu thuộc lớp i.

        - $c$: Số lớp.
        """)

        st.write("""
        #### Entropy và Information Gain (Phân loại)

        - **Entropy**:
        """)

        st.latex(r"\text{Entropy} = - \sum_{i=1}^{c} p_i \log_2(p_i)")

        st.write("- **Information Gain**:")

        st.latex(
            r"\text{IG} = \text{Entropy(parent)} - \sum_{j} \frac{N_j}{N} \text{Entropy(child}_j\text{)}")

        st.write("#### Mean Squared Error (Hồi quy)")

        st.latex(r"\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2")

        st.markdown(r"""
        - $y_i$: Giá trị thực tế.
                    
        - $\bar{y}$: Giá trị trung bình.
                    
        """)

    # # Phần 4: Ví dụ minh họa
    # st.subheader("Ví dụ minh họa")
    # st.write("""
    # Giả sử dữ liệu có 2 đặc trưng: "Tuổi" và "Thu nhập", nhãn là "Mua nhà" (Có/Không):
    # - Bước 1: Tính Gini cho toàn bộ dữ liệu.
    # - Bước 2: Chia với "Tuổi ≤ 30":
    #     - Nhánh ≤ 30: 80% Không, 20% Có.
    #     - Nhánh > 30: 60% Có, 40% Không.
    # - Bước 3: Chọn "Tuổi ≤ 30" làm nút gốc nếu Gini giảm nhiều.
    # - Bước 4: Tiếp tục chia trên nhánh với "Thu nhập".
    # """)

    with col1:
        # Phần 5: Ưu và nhược điểm
        st.subheader("Ưu và nhược điểm")
        st.write("""
        **Ưu điểm**:
        - Dễ hiểu, trực quan.
        - Không cần chuẩn hóa dữ liệu.
        - Xử lý được cả dữ liệu số và phân loại.

        **Nhược điểm**:
        - Dễ bị overfitting nếu không giới hạn độ sâu.
        - Nhạy cảm với nhiễu và dữ liệu mất cân bằng.
        - Không tốt với mối quan hệ phi tuyến phức tạp.
        """)


def svm_theory():
    pass


def theory_info():
    st.title("Thông tin về các thuật toán")
    st.markdown("""
    - Decision Tree: Thuật toán dự đoán giá trị đầu ra dựa trên các cây quyết định.
    - Support Vector Machine (SVM): Thuật toán học máy tính đặc trưng (SVM) cho phân lớp hai hoặc nhiều lớp.
    """)
