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

    with open('./services/mnist_classfier/data/X.pkl', 'rb') as f:
        X = pickle.load(f)

    with open('./services/mnist_classfier/data/y.pkl', 'rb') as f:
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

        # st.write("#### Mean Squared Error (Hồi quy)")

        # st.latex(r"\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2")

        # st.markdown(r"""
        # - $y_i$: Giá trị thực tế.

        # - $\bar{y}$: Giá trị trung bình.

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
    st.markdown(r"""
    ## Support Vector Machine - SVM

    ### 1. Tổng quan

    **Máy Vector Hỗ trợ (SVM)** là một thuật toán học máy có giám sát, được sử dụng chủ yếu cho các bài toán phân loại (classification) và có thể mở rộng cho hồi quy (regression). SVM được phát triển bởi Vladimir Vapnik và các cộng sự vào những năm 1990. Ý tưởng chính của SVM là tìm một **siêu phẳng (hyperplane)** phân tách tốt nhất giữa các lớp dữ liệu, sao cho khoảng cách từ siêu phẳng đến các điểm dữ liệu gần nhất của mỗi lớp (gọi là **support vectors**) là lớn nhất.

    Một số khái niệm cơ bản trong SVM:

    - **Siêu phẳng**: Trong không gian $n$-chiều, siêu phẳng được định nghĩa bởi phương trình """)

    st.latex(r"w^T x + b = 0 ")

    st.markdown(r"""
    Trong đó:

    $w$ là vector pháp tuyến

    $x$ là điểm trong không gian

    $b$ là hệ số tự do.

    - Ví dụ: Trong không gian 2 chiều, siêu phẳng là một đường thẳng; trong không gian 3 chiều, siêu phẳng là một mặt phẳng.
    - **Support Vectors**: Các điểm dữ liệu nằm gần nhất với siêu phẳng phân tách, đóng vai trò quan trọng trong việc xác định vị trí và hướng của siêu phẳng.
    - **Margin**: Khoảng cách từ siêu phẳng đến support vectors. SVM tìm siêu phẳng sao cho margin này là lớn nhất.

    SVM có thể xử lý cả dữ liệu **tuyến tính phân tách** và **không tuyến tính phân tách** nhờ sử dụng **kernel trick**, giúp ánh xạ dữ liệu lên không gian chiều cao hơn để phân tách dễ dàng hơn.

    ---

    ### 2. Nguyên lý hoạt động

    SVM hoạt động dựa trên việc tối ưu hóa siêu phẳng phân tách giữa các lớp dữ liệu. Có hai trường hợp chính: dữ liệu **tuyến tính phân tách** và **không tuyến tính phân tách**.

    #### 2.1. Trường hợp tuyến tính phân tách (Hard Margin SVM)

    **Mục tiêu**: Tìm siêu phẳng phân tách sao cho **margin** giữa các lớp là lớn nhất.

    Giả sử tập dữ liệu $(x_i, y_i)_{i=1}^N$, với $x_i \in \mathbb{R}^n$ là vector đặc trưng và $y_i \in \{-1, 1\}$ là nhãn lớp.
    Siêu phẳng phân tách được biểu diễn bởi:""")

    st.latex(r"w^T x + b = 0")

    st.markdown(r"Để phân tách đúng, các điểm dữ liệu phải thỏa mãn:")

    st.latex(r"y_i (w^T x_i + b) \geq 1 \quad \forall i")

    st.markdown(r"**Margin** được tính bằng:")

    st.latex(r"\text{Margin} = \frac{2}{\|w\|}")

    st.markdown(
        r"SVM tìm $w$ và $b$ để tối đa hóa margin, tức là tối thiểu hóa $\frac{1}{2} \|w\|^2$, với ràng buộc:")

    st.latex(r"\text{Minimize} \quad \frac{1}{2} \|w\|^2")

    st.latex(
        r"\text{Subject to} \quad y_i (w^T x_i + b) \geq 1 \quad \forall i")

    st.markdown(r"""

    #### 2.2. Trường hợp không tuyến tính phân tách (Soft Margin SVM)

    Khi dữ liệu không thể phân tách tuyến tính hoàn toàn, SVM sử dụng **Soft Margin** để cho phép một số điểm bị phân
    loại sai hoặc nằm trong margin. SVM các biến slack $\xi_i \geq 0$ và điều chỉnh bài toán tối ưu:""")

    st.latex(
        r"\text{Minimize} \quad \frac{1}{2} \|w\|^2 + C \sum_{i=1}^N \xi_i")

    st.latex(
        r"\text{Subject to} \quad y_i (w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0 \quad \forall i")

    st.markdown(r"""

    Trong đó $C$ là tham số điều chỉnh, cân bằng giữa tối đa hóa margin và giảm thiểu lỗi phân loại.
                
    $C$ lớn: Ưu tiên giảm lỗi phân loại, dẫn đến margin nhỏ hơn (ít khoan dung với lỗi).
                
    $C$ nhỏ: Ưu tiên tối đa hóa margin, chấp nhận nhiều lỗi hơn.
                
    **Biến slack $\xi_i$** đo lường mức độ vi phạm của điểm dữ liệu $x_i$ đối với điều kiện margin:
    - $\xi_i = 0$: Điểm $x_i$ được phân loại đúng và nằm ngoài margin (hoặc trên ranh giới margin).
    - $0 < \xi_i \leq 1 $: Điểm $x_i$ nằm trong margin nhưng vẫn được phân loại đúng.
    - $\xi_i > 1$: Điểm $x_i$ bị phân loại sai (nằm ở phía sai của siêu phẳng phân tách).

    Biến slack giúp SVM linh hoạt hơn, cho phép xử lý dữ liệu có nhiễu hoặc các điểm ngoại lai, đồng thời cân bằng giữa việc tối đa hóa margin và giảm thiểu lỗi phân loại thông qua tham số $C$.

    #### 2.3. Dữ liệu không tuyến tính (Kernel Trick)

    Khi dữ liệu không thể phân tách tuyến tính trong không gian ban đầu, SVM sử dụng **kernel trick** để ánh xạ dữ liệu lên không gian chiều cao hơn. Một số kernel phổ biến:

    - **Linear Kernel**: $K(x_i, x_j) = x_i^T x_j$
    - **Polynomial Kernel**: $K(x_i, x_j) = (x_i^T x_j + 1)^d$
    - **RBF Kernel (Gaussian)**: $K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$

    Bài toán tối ưu được chuyển sang dạng đối ngẫu:""")

    st.latex(r"\text{Maximize} \quad \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j K(x_i, x_j)")

    st.latex(
        r"\text{Subject to} \quad \sum_{i=1}^N \alpha_i y_i = 0, \quad 0 \leq \alpha_i \leq C \quad \forall i")

    st.markdown(
        r"""Siêu phẳng phân tách được xây dựng dựa trên các support vectors (các điểm có $\alpha_i > 0$)
    """)

    st.markdown(r"""
    ### 3. Ưu điểm của SVM
    
    - Hiệu quả cao: SVM hoạt động rất tốt trên các bài toán phân loại, đặc biệt khi dữ liệu có số chiều lớn (như MNIST).
    
    - Khả năng tổng quát hóa tốt: Nhờ tối đa hóa margin, SVM ít bị overfitting (nếu tham số C được chọn phù hợp).
    
    - Linh hoạt với kernel: Có thể xử lý dữ liệu không tuyến tính phân tách bằng cách sử dụng kernel trick.
    
    - Chỉ phụ thuộc vào support vectors: SVM không bị ảnh hưởng bởi các điểm dữ liệu xa siêu phẳng, giúp giảm nhạy cảm với nhiễu.
""")

    st.markdown(r"""
    ### 4. Nhược điểm của SVM
                
    - Tốn tài nguyên tính toán: SVM có độ phức tạp tính toán cao không phù hợp với tập dữ liệu rất lớn.
    
    - Nhạy cảm với tham số: Hiệu suất của SVM phụ thuộc nhiều vào việc chọn C, gamma, và loại kernel.
    
    - Khó xử lý dữ liệu không chuẩn hóa: SVM nhạy cảm với tỷ lệ của dữ liệu, cần chuẩn hóa (scaling) trước khi huấn luyện.
""")


def full_theory():
    option = st.selectbox(
        'Chọn phương pháp giải thích:',
        ('DecisionTree', 'SVM')
    )

    if option == 'DecisionTree':
        decision_tree_theory()
    elif option == 'SVM':
        svm_theory()


def theory_info():
    st.title("Thông tin về các thuật toán")
    st.markdown("""
    - Decision Tree: Thuật toán dự đoán giá trị đầu ra dựa trên các cây quyết định.
    - Support Vector Machine(SVM): Thuật toán học máy tính đặc trưng(SVM) cho phân lớp hai hoặc nhiều lớp.
    """)
