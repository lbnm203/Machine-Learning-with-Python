import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Hàm giảm chiều dữ liệu bằng PCA


def reduce_dimension(X, n_components):
    """Giảm chiều dữ liệu xuống n_components chiều bằng PCA."""
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced

# Hàm huấn luyện Kmeans


def train_kmeans(X, n_clusters, init, max_iter):
    """Huấn luyện mô hình Kmeans với số cụm n_clusters."""
    kmeans = KMeans(n_clusters=n_clusters, init=init, max_iter=max_iter, random_state=42)
    labels = kmeans.fit_predict(X)
    return labels

# Hàm train DBSCAN


def train_dbscan(X, eps, min_samples, metrics, algorithm):
    """Huấn luyện mô hình DBSCAN với tham số eps và min_samples."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metrics=metrics, algorithm=algorithm)
    labels = dbscan.fit_predict(X)
    return labels

# Hàm vẽ biểu đồ scatter plot 2D


def plot_clusters(X_2d, dbscan_labels, kmeans_labels):
    """Vẽ biểu đồ scatter plot 2D cho kết quả phân cụm của DBSCAN và Kmeans."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Biểu đồ DBSCAN
    ax[0].scatter(X_2d[:, 0], X_2d[:, 1], c=dbscan_labels, cmap='viridis', s=5)
    ax[0].set_title("Phân cụm DBSCAN")

    # Biểu đồ Kmeans
    ax[1].scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap='viridis', s=5)
    ax[1].set_title("Phân cụm Kmeans")

    return fig


def train_process(X, y):
    st.write("Quá trình huấn luyện")

    total_samples = X.shape[0]

    # Chọn số lượng ảnh để train
    num_samples = st.number_input(
        'Chọn số lượng ảnh cho phần huấn luyện', 100, total_samples, 20000)

    # Chọn số lượng ảnh theo yêu cầu
    X_selected, y_selected = X[:num_samples], y[:num_samples]

    # Chọn tỷ lệ train và test
    test_size = st.slider('Test size', 0.0, 1.0, 0.3)

    # Chia train/test theo tỷ lệ đã chọn
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y_selected, test_size=test_size, random_state=42)

    # Hiển thị số lượng mẫu dưới dạng bảng
    st.write("### Số lượng mẫu")
    data = {
        "Tập": ["Train", "Test"],
        "Số lượng mẫu": [X_train.shape[0], X_test.shape[0]],
        "Tỷ lệ (%)": [int((1 - test_size) * 100), int(test_size * 100)]
    }
    st.table(data)

    # Lưu vào session state
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test

    # Chuẩn hóa dữ liệu
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    # Chọn có dùng PCA không
    use_pca = st.checkbox("Sử dụng PCA để giảm chiều dữ liệu", value=True)
    if use_pca:
        n_components = st.slider("Số chiều sau PCA",
                                 2, 100, 50,
                                 help="""PCA (Principal Component Analysis) là một kỹ thuật giảm chiều dữ liệu trong học máy và thống kê, 
                                 giúp biến đổi dữ liệu từ không gian nhiều chiều sang không gian ít chiều hơn mà vẫn giữ lại được phần lớn 
                                 thông tin quan trọng. PCA có thể loại bỏ nhiễu và các biến dư thừa, giúp cải thiện hiệu suất của các mô hình học máy.""")
        X_train_reduced = reduce_dimension(X_train, n_components)
    else:
        X_train_reduced = X_train

    st.write("---")

    col1, col2 = st.columns(2)
    with col1:
        # Chọn thuật toán phân cụm
        model_type = st.selectbox(
            'Chọn Thuật Toán Phân Cụm', ('KMeans', 'DBSCAN'))
        st.write('---')

        if model_type == "KMeans":
            n_clusters = st.slider("Số cụm (K)", 2, 20, 10, help="Số lượng tâm cần tạo ra")
            init = st.selectbox("Init", ('k-means++', 'random'), help="""
- `k-means++`: chọn trọng tâm cụm ban đầu bằng cách lấy mẫu dựa trên phân phối xác suất thực nghiệm của sự đóng góp 
của các điểm vào quán tính tổng thể. Kỹ thuật này tăng tốc độ hội tụ.
- `random`: chọn `n_clusters` ngẫu nhiên các quan sát (hàng) từ dữ liệu cho trọng tâm ban đầu.                                
""")
            max_iter = st.slider("Max iterations", 100, 1000, 300, help="Số lần lặp tối đa của thuật toán k-means cho một lần chạy duy nhất")
            if st.button("Huấn luyện KMeans"):
                kmeans_labels = train_kmeans(X_train_reduced, n_clusters, init, max_iter)
                st.session_state["kmeans_labels"] = kmeans_labels
                st.success("Huấn luyện KMeans thành công!")

                st.markdown("Silhouette Score", help="""
- Silhouette Score là một chỉ số đánh giá chất lượng của các cụm (clusters) trong phân cụm dữ liệu. 
Nó đo lường mức độ tương đồng của một điểm dữ liệu với các điểm trong cùng cụm so với các điểm trong 
các cụm khác. Giá trị Silhouette Score nằm trong khoảng từ -1 đến 1, trong đó:
    - Giá trị càng gần 1: Các cụm càng tốt, điểm dữ liệu được gán đúng cụm.
    - Giá trị gần 0: Điểm dữ liệu nằm gần ranh giới giữa hai cụm.
    - Giá trị âm: Điểm dữ liệu có thể bị gán sai cụm.                           
""")
                # Đánh giá KMeans
                if len(np.unique(kmeans_labels)) > 1:
                    silhouette_kmeans = silhouette_score(
                        X_train_reduced, kmeans_labels)

                    st.success(f"Silhouette Score: {silhouette_kmeans:.4f}")
                else:
                    st.warning(
                        "Không thể tính Silhouette Score (chỉ có 1 cụm)")

                with col2:
                    # Giảm chiều xuống 2D để vẽ
                    if use_pca and n_components != 2:
                        X_train_2d = reduce_dimension(X_train, 2)
                    else:
                        X_train_2d = X_train_reduced

                    # Vẽ biểu đồ
                    st.header("Kết quả phân cụm KMeans")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1],
                               c=kmeans_labels, cmap='viridis', s=5)
                    ax.set_title("Phân cụm KMeans")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    st.pyplot(fig)

        elif model_type == "DBSCAN":
            st.markdown("""
            - **Epsilon (ε)**: giúp xác định các điểm nằm trong vùng lân cận epsilon.""", 
            help="""
- **Epsilon (ε)** thấp có thể dẫn đến:
    - vùng lân cận của mỗi điểm rất nhỏ, dẫn đến ít điểm được coi là lân cận của nhau
    - Nhiều điểm có thể bị coi là nhiễu (noise) vì không đủ điểm trong vùng lân cận
    - Các cụm có thể bị chia nhỏ thành nhiều cụm nhỏ hoặc thậm chí không hình thành cụm nào.
        
    → Số lượng cụm tăng lên, nhiều điểm bị gán nhãn là nhiễu.
- **Epsilon (ε)** cao có thể dẫn đến:
    - Vùng lân cận của mỗi điểm rất lớn, dẫn đến nhiều điểm được coi là lân cận của nhau
    - Các cụm có thể bị hợp nhất thành một cụm lớn, ngay cả khi chúng không thực sự thuộc cùng một cụm.
        
    → Số lượng cụm giảm xuống, có thể chỉ còn một cụm duy nhất, ít điểm bị gán nhãn là nhiễu.""")
            
            st.markdown("""
            - **Min_samples**: Số lượng điểm tối thiểu trong vùng lân cận ε để một điểm được coi là điểm lõi (core point).""",
            help="""
- **Min_samples** thấp có thể dẫn đến:
    - Rất dễ để một điểm trở thành điểm lõi, ngay cả khi nó nằm trong vùng có mật độ thấp.
    - Các cụm có thể chứa nhiều điểm nhiễu hoặc không đồng nhất.
    → Số lượng cụm tăng lên, các cụm có thể không chính xác và chứa nhiều nhiễu. 
- **Min_samples** cao có thể dẫn đến:
    - Khó để một điểm trở thành điểm lõi, vì cần nhiều điểm trong vùng lân cận.
    - Các cụm có thể bị bỏ sót, đặc biệt là các cụm nhỏ hoặc có mật độ thấp.
    → Số lượng cụm giảm xuống, nhiều điểm bị gán nhãn là nhiễu, ngay cả khi chúng thuộc về một cụm thực sự.
""")
            st.markdown("""
            - **metric**: Hàm khoảng cách để đo lường khoảng cách giữa hai điểm bất kì""",
            help="""mặc định là euclidean""")

            st.markdown("""
            - **algorithm**: phương pháp được sử dụng để xác định các điểm láng giềng""",
            help="""Bao gồm các phương pháp auto, ball_tree, kd_tree, brute: 
- `auto`: DBSCAN tự động chọn thuật toán tối ưu dựa trên đặc điểm của dữ liệu (ví dụ: số chiều, số lượng điểm dữ liệu).
Nó sẽ ưu tiên sử dụng kd_tree hoặc ball_tree nếu dữ liệu phù hợp, ngược lại sẽ sử dụng brute.
- `ball_tree`: Sử dụng cấu trúc dữ liệu Ball Tree để tổ chức các điểm dữ liệu trong không gian.
Ball Tree chia không gian thành các hình cầu (balls) lồng nhau, giúp tìm kiếm lân cận hiệu quả hơn trong không gian nhiều chiều. 
Sử dụng khi dữ liệu có số chiều cao (high-dimensional data) và kd_tree không hiệu quả.
- `kd_tree`: Sử dụng cấu trúc dữ liệu KD-Tree (K-dimensional Tree) để tổ chức các điểm dữ liệu.
KD-Tree chia không gian thành các vùng hình chữ nhật dựa trên các trục tọa độ. Sử dụng khi dữ liệu có số chiều thấp đến trung bình 
(thường dưới 20 chiều).
- `brute`: Sử dụng phương pháp vét cạn (brute-force) để tính toán khoảng cách giữa tất cả các cặp điểm.
Không sử dụng bất kỳ cấu trúc dữ liệu nào để tối ưu hóa tìm kiếm. Sử dụng khi dữ liệu có số chiều rất cao hoặc khi các thuật toán khác không hiệu quả.
""")

            eps = st.slider("Epsilon", 0.1, 5.0, 1.0)
            min_samples = st.slider("Min Samples", 2, 20, 5)
            metrics = st.selectbox("Metric", ["euclidean", "manhattan"])
            algorithm = st.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])
            if st.button("Huấn luyện DBSCAN"):
                dbscan_labels = train_dbscan(X_train_reduced, eps, min_samples, metrics, algorithm)
                st.session_state["dbscan_labels"] = dbscan_labels
                st.success("Huấn luyện DBSCAN thành công!")

                st.markdown("Silhouette Score", help="""
- Silhouette Score là một chỉ số đánh giá chất lượng của các cụm (clusters) trong phân cụm dữ liệu. 
Nó đo lường mức độ tương đồng của một điểm dữ liệu với các điểm trong cùng cụm so với các điểm trong 
các cụm khác. Giá trị Silhouette Score nằm trong khoảng từ -1 đến 1, trong đó:
    - Giá trị càng gần 1: Các cụm càng tốt, điểm dữ liệu được gán đúng cụm.
    - Giá trị gần 0: Điểm dữ liệu nằm gần ranh giới giữa hai cụm.
    - Giá trị âm: Điểm dữ liệu có thể bị gán sai cụm.                           
""")
                # Đánh giá DBSCAN
                if len(np.unique(dbscan_labels)) > 1:
                    mask = dbscan_labels != -1
                    X_no_noise = X_train_reduced[mask]
                    labels_no_noise = dbscan_labels[mask]
                    if len(np.unique(labels_no_noise)) > 1:
                        silhouette_dbscan = silhouette_score(
                            X_no_noise, labels_no_noise)
                        
                        st.success(
                            f"Silhouette Score (không tính nhiễu): {silhouette_dbscan:.4f}")
                    else:
                        st.warning(
                            "Không thể tính Silhouette Score (chỉ có 1 cụm sau khi loại nhiễu)")
                else:
                    st.warning(
                        "Không thể tính Silhouette Score (chỉ có 1 cụm hoặc toàn nhiễu)")

                with col2:
                    # Giảm chiều xuống 2D để vẽ
                    if use_pca and n_components != 2:
                        X_train_2d = reduce_dimension(X_train, 2)
                    else:
                        X_train_2d = X_train_reduced

                    # Vẽ biểu đồ
                    st.write("#### Kết quả phân cụm DBSCAN")
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.scatter(X_train_2d[:, 0], X_train_2d[:, 1],
                               c=dbscan_labels, cmap='viridis', s=5)
                    ax.set_title("Phân cụm DBSCAN")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    st.pyplot(fig)