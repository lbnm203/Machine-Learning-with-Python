import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import os
import mlflow
import time
import datetime


def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/lbnm203/Machine_Learning_UI.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "lbnm203"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "0902d781e6c2b4adcd3cbf60e0f288a8085c5aab"

    mlflow.set_experiment("MNIST_Clustering")


@st.cache_resource
def train_kmeans(X, n_clusters):
    # Đảm bảo X là 2D
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X)
    return kmeans, labels


@st.cache_resource
def train_dbscan(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)
    return dbscan, labels


def plot_clusters(X, labels, method, params, run):
    # Vì không dùng PCA, ta sẽ vẽ một biểu đồ đơn giản hơn, ví dụ: histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    unique_labels = np.unique(labels)
    ax.hist(labels, bins=len(unique_labels), edgecolor='black')
    ax.set_title(f"{method} Cluster Distribution")
    ax.set_xlabel("Cluster Label")
    ax.set_ylabel("Number of Samples")
    st.pyplot(fig)

    plot_file = f"{method.lower()}_distribution.png"
    fig.savefig(plot_file)
    if run:
        mlflow.log_artifact(plot_file, artifact_path="visualizations")
    plt.close(fig)


def visualize_dbscan(X, labels, images):
    unique_labels = np.unique(labels)
    # Đếm số cụm, không tính noise
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)

    # 1. Hiển thị lưới hình ảnh đại diện
    n_samples_per_cluster = 5
    fig, axes = plt.subplots(len(unique_labels), n_samples_per_cluster,
                             figsize=(n_samples_per_cluster*2, len(unique_labels)*2))
    for i, cluster in enumerate(unique_labels):
        cluster_indices = np.where(labels == cluster)[0]
        samples = np.random.choice(cluster_indices, min(
            n_samples_per_cluster, len(cluster_indices)), replace=False)
        for j, sample_idx in enumerate(samples):
            if len(unique_labels) == 1:
                ax = axes[j] if n_samples_per_cluster > 1 else axes
            else:
                ax = axes[i, j]
            ax.imshow(images[sample_idx], cmap='gray')
            ax.axis('off')
            if j == 0:
                ax.set_title(
                    f'{"Noise" if cluster == -1 else f"Cụm {cluster}"}')
    plt.suptitle(
        f"Kết quả DBSCAN: {n_clusters} cụm, {n_noise} điểm noise", y=1.05)
    plt.tight_layout()

    # 2. Biểu đồ cột số lượng mẫu
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    cluster_counts.plot(kind='bar', ax=ax2)
    ax2.set_title("Số lượng mẫu trong mỗi cụm (bao gồm noise)")
    ax2.set_xlabel("Nhãn cụm (-1 là noise)")
    ax2.set_ylabel("Số lượng mẫu")
    plt.tight_layout()

    return fig, fig2


def compare_clusters_with_true_labels(X, y_true, labels_pred, method, n_samples_per_cluster=5):
    """
    So sánh nhãn cụm dự đoán với nhãn thật và hiển thị ảnh mẫu.
    X: Dữ liệu gốc (chưa reshape)
    y_true: Nhãn thật
    labels_pred: Nhãn cụm dự đoán
    method: Tên mô hình (KMeans hoặc DBSCAN)
    n_samples_per_cluster: Số ảnh mẫu hiển thị cho mỗi cụm
    """
    # Hiển thị ảnh mẫu từ mỗi cụm (bao gồm nhiễu nếu có)
    unique_labels = np.unique(labels_pred)
    st.write(f"### Hiển thị {n_samples_per_cluster} ảnh mẫu từ mỗi cụm/nhiễu")

    for cluster in unique_labels:
        if cluster == -1 and method == "DBSCAN":
            st.write(f"#### Nhiễu (Cluster -1)")
        else:
            st.write(f"#### Cụm {cluster}")

        # Lấy chỉ số của các mẫu trong cụm này
        cluster_indices = np.where(labels_pred == cluster)[0]
        if len(cluster_indices) == 0:
            st.write("Không có mẫu nào trong cụm này.")
            continue

        # Chọn ngẫu nhiên tối đa n_samples_per_cluster mẫu
        sample_indices = np.random.choice(cluster_indices,
                                          min(n_samples_per_cluster,
                                              len(cluster_indices)),
                                          replace=False)

        # Tạo figure để hiển thị ảnh
        fig_kmeans, axes = plt.subplots(
            1, len(sample_indices), figsize=(2 * len(sample_indices), 2))
        if len(sample_indices) == 1:
            axes = [axes]  # Đảm bảo axes là iterable

        for idx, ax in zip(sample_indices, axes):
            # Giả sử X là ảnh MNIST (28x28), cần reshape lại từ flatten nếu cần
            img = X[idx].reshape(28, 28) if X[idx].size == 784 else X[idx]
            ax.imshow(img, cmap='gray')
            ax.axis('off')

            # Save the figure to a file only if within an active MLflow run
            if mlflow.active_run():
                plot_file = f"{method.lower()}_cluster_samples.png"
                fig_kmeans.savefig(plot_file)
                mlflow.log_artifact(plot_file, artifact_path="visualizations")

                # Lưu vào session_state
                if "cluster_fig" not in st.session_state:
                    st.session_state["cluster_fig"] = {}

                st.session_state["cluster_fig"]["KMeans"] = fig_kmeans

        st.pyplot(fig_kmeans)
        plt.close(fig_kmeans)


def train_process(X, y):
    st.write("Quá trình huấn luyện")

    total_samples = X.shape[0]

    num_samples = st.slider(
        'Chọn số lượng ảnh cho phần huấn luyện', 1000, total_samples, 10000)
    X_selected, y_selected = X[:num_samples], y[:num_samples]

    test_size = st.slider('Test size', 0.0, 1.0, 0.3)

    if st.button("Chia Dữ Liệu"):
        with st.spinner("Đang tải và chia dữ liệu..."):
            start_time = time.time()
            X_train, X_test, y_train, y_test = train_test_split(
                X_selected, y_selected, test_size=test_size, random_state=42)
            training_time = time.time() - start_time

            st.success(f"👉 Chia Dữ Liệu Thành Công! - {training_time:.2f}s")

            st.write("### Số lượng mẫu")
            data = {
                "Tập": ["Train", "Test"],
                "Số lượng mẫu": [X_train.shape[0], X_test.shape[0]],
                "Tỷ lệ (%)": [int((1 - test_size) * 100), int(test_size * 100)]
            }
            st.table(data)

            st.session_state['X_train'] = X_train
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_test'] = y_test

    if "X_train" not in st.session_state:
        st.warning("⚠️ Vui lòng chia dữ liệu trước khi train!")
        return

    X_train = st.session_state["X_train"]
    y_train = st.session_state["y_train"]

    # Chuẩn hóa dữ liệu
    X_train_norm = (X_train.reshape(-1, 28 * 28) / 255.0).astype(np.float32)
    st.session_state["X_train_norm"] = X_train_norm

    model_choice = st.selectbox("Chọn mô hình:", ["KMeans", "DBSCAN"])

    mlflow_input()
    run_name = st.text_input("Enter Run Name:", "")
    if run_name.strip() == "" or run_name.strip() == " ":
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_name = f"MNIST_Clustering_{timestamp.replace(' ', '_').replace(':', '-')}"

    st.session_state["run_name"] = run_name

    if model_choice == "KMeans":
        n_clusters = st.slider("🔢 Chọn số cụm (K):", 2, 20, 10)
        # n_init = st.slider("🔄 Số lần khởi tạo:", 1, 20, 10)

        if st.button("Train KMeans"):
            with mlflow.start_run(run_name=run_name):
                with st.spinner("Đang huấn luyện KMeans..."):
                    start_time = time.time()
                    model, labels = train_kmeans(
                        X_train_norm, n_clusters)
                    training_time = time.time() - start_time

                    # Tính silhouette score
                    silhouette = silhouette_score(X_train_norm, labels)

                    # Log parameters và metrics vào MLflow
                    mlflow.log_param("model", "KMeans")
                    mlflow.log_param("Training Size", X_train.shape[0])
                    mlflow.log_param("n_clusters", n_clusters)
                    mlflow.log_metric("silhouette_score", silhouette)
                    mlflow.log_metric("training_time", training_time)

                    col1, col2, col3 = st.columns([1, 3, 1])
                    # Vẽ phân bố cụm
                    with col2:
                        plot_clusters(X_train_norm, labels, "KMeans",
                                      {"n_clusters": n_clusters}, True)
                    compare_clusters_with_true_labels(
                        X_train, y_train, labels, "KMeans")

                    st.success(
                        f"✅ Huấn luyện KMeans hoàn tất! Thời gian: {training_time:.2f}s")
                    st.success(f"Silhouette Score: {silhouette:.4f}")

            model_name = model_choice.lower().replace(" ", "_")
            count = 1
            new_model_name = model_name
            while any(m["name"] == new_model_name for m in st.session_state["models"]):
                new_model_name = f"{model_name}_{count}"
                count += 1

            st.session_state["models"].append(
                {"name": new_model_name, "model": model})
            st.write(f"🔹 **Mô hình đã được lưu với tên:** `{new_model_name}`")

    elif model_choice == "DBSCAN":
        eps = st.slider("Bán kính lân cận (epsilon):", 0.1, 10.0, 0.5)
        min_samples = st.slider("Số điểm tối thiểu trong cụm:", 2, 20, 5)

        if st.button("Train DBSCAN"):
            with mlflow.start_run(run_name=run_name):
                with st.spinner("Đang huấn luyện DBSCAN..."):
                    start_time = time.time()
                    model, labels = train_dbscan(
                        X_train_norm, eps, min_samples)
                    training_time = time.time() - start_time

                    # Đếm số cụm (loại bỏ nhiễu -1)
                    n_clusters = len(set(labels) - {-1})
                    silhouette = silhouette_score(
                        X_train_norm, labels) if n_clusters > 1 else -1
                    ari = adjusted_rand_score(y_train, labels)
                    noise_ratio = np.sum(labels == -1) / len(labels)
                    # Log parameters và metrics
                    mlflow.log_param("Model", "DBSCAN")
                    mlflow.log_param("eps", eps)
                    mlflow.log_param("min_samples", min_samples)
                    mlflow.log_metric("n_clusters", n_clusters)
                    mlflow.log_metric("Silhouette_score", silhouette)
                    mlflow.log_metric("noise_ratio", noise_ratio)
                    mlflow.log_metric("Training_time", training_time)

                    col1, col2, col3 = st.columns([1, 3, 1])
                    # Vẽ phân bố cụm
                    with col2:
                        # Vẽ phân bố cụm
                        visualize_dbscan()
                    #     plot_clusters(X_train_norm, labels, "DBSCAN",
                    #                   {"eps": eps, "min_samples": min_samples}, True)
                    # compare_clusters_with_true_labels(
                    #     X_train, y_train, labels, "DBSCAN")

                    st.success(
                        f"✅ Huấn luyện DBSCAN hoàn tất! Thời gian: {training_time:.2f}s")
                    st.success(f"Số cụm tìm được: {n_clusters}")
                    st.success(f"Tỷ lệ nhiễu: {noise_ratio}")
                    st.success(f"Silhouette Score: {silhouette:.4f}")

            model_name = model_choice.lower().replace(" ", "_")
            count = 1
            new_model_name = model_name
            while any(m["name"] == new_model_name for m in st.session_state["models"]):
                new_model_name = f"{model_name}_{count}"
                count += 1

            st.session_state["models"].append(
                {"name": new_model_name, "model": model})
            st.write(f"🔹 **Mô hình đã được lưu với tên:** `{new_model_name}`")
