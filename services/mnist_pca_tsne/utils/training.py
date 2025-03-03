import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
import os
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import numpy as np
import time


def input_mlflow():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/lbnm203/Machine_Learning_UI.mlflow/"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "lbnm203"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "0902d781e6c2b4adcd3cbf60e0f288a8085c5aab"
    mlflow.set_experiment("MNIST_PCA_t-SNE")


@st.cache_data
def fit_tnse(X, n_components, perplexity, learning_rate, n_iter, metric):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate,
                n_iter=n_iter, metric=metric, random_state=42)
    X_train_tsne = tsne.fit_transform(X)
    return X_train_tsne, tsne


def train_pca(X, y):
    total_samples = X.shape[0]

    # Chọn số lượng ảnh để train
    num_samples = st.slider(
        'Chọn số lượng ảnh cho phần huấn luyện', 1000, total_samples, 70000)

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

    if "X_train" not in st.session_state or "X_test" not in st.session_state or "y_train" not in st.session_state or "y_test" not in st.session_state:
        st.session_state["X_train"] = X_train
        st.session_state["X_test"] = X_test
        st.session_state["y_train"] = y_train
        st.session_state["y_test"] = y_test

    # Chuẩn hóa dữ liệu
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    input_mlflow()

    run_name = st.text_input("Nhập tên Run:", "Default Run")
    if not run_name:
        run_name = "default_run"
    st.session_state["run_name"] = run_name

    dim_reduction_method = st.selectbox(
        "**Chọn phương pháp rút gọn chiều dữ liệu:**", ["PCA", "t-SNE"])
    if dim_reduction_method == "PCA":
        col1, col2 = st.columns(2)
        with col1:
            # Tham số của PCA
            n_components = st.slider("**Số thành phần chính (n_components):**",
                                     min_value=2, max_value=min(X_train.shape[1], 20),
                                     value=5,
                                     help="""
Số lượng chiều (`n_components`) muốn giữ lại sau khi giảm chiều bằng PCA. 
- Giá trị nhỏ hơn (ví dụ: 2-5) phù hợp cho trực quan hóa, nhưng có thể mất thông tin. 
- Giá trị lớn hơn giữ lại nhiều thông tin hơn nhưng làm tăng độ phức tạp tính toán
""")
            svd_solver = st.selectbox("**Thuật toán SVD:**",
                                      ["auto", "full", "arpack", "randomized"],
                                      help="""
Thuật toán SVD được sử dụng để tính toán PCA:
- `auto`: Chọn thuật toán tự động dựa trên kích thước dữ liệu.
- `full`: Tính toàn bộ ma trận hiệp phương sai, phù hợp với dữ liệu nhỏ.
- `arpack`: Dùng cho dữ liệu lớn, tối ưu hóa bộ nhớ nhưng chậm hơn.
- `randomized`: Tăng tốc trên dữ liệu lớn, nhưng có thể ít chính xác hơn với dữ liệu nhỏ.
                """)

            st.warning("""
- Thứ tự ưu tiên :
    - "auto" (Mặc định, ưu tiên cao nhất):
        - Lý do: Tự động chọn thuật toán tốt nhất dựa trên dữ liệu, không cần can thiệp thủ công, phù hợp với hầu hết các trường hợp.
    - "randomized" (Ưu tiên cao, đặc biệt với dữ liệu lớn):
        - Lý do: Nhanh nhất trên dữ liệu lớn, phù hợp với MNIST hoặc các tập dữ liệu có hàng ngàn mẫu và đặc trưng.
    - "arpack" (Ưu tiên trung bình, phù hợp với dữ liệu lớn nhưng cần kiểm soát bộ nhớ):
        - Lý do: Tối ưu hóa bộ nhớ, nhưng chậm hơn "randomized" và chỉ hữu ích khi n_features >> n_samples.
    - "full" (Ưu tiên thấp nhất, chỉ dùng với dữ liệu nhỏ):
        - Lý do: Chính xác nhất nhưng tốn bộ nhớ và thời gian, không phù hợp với dữ liệu lớn.
""")

        if st.button("🚀 Chạy PCA"):
            progress_bar = st.progress(0)
            for i in range(1, 101):
                progress_bar.progress(i)
                time.sleep(0.01)
            st.write("Quá trình huấn luyện đã hoàn thành!")
            with mlflow.start_run(run_name=st.session_state["run_name"]) as run:
                mlflow.set_tag("mlflow.runName", st.session_state["run_name"])
                # Áp dụng PCA
                pca = PCA(n_components=n_components,
                          svd_solver=svd_solver, random_state=42)
                X_train_pca = pca.fit_transform(X_train)

                progress_bar = st.progress(0)
                for i in range(1, 101):
                    progress_bar.progress(i)
                    time.sleep(0.01)
                st.write("Quá trình huấn luyện đã hoàn thành!")

                st.session_state.X_train_pca = X_train_pca
                st.session_state.explained_variance_ratio_ = pca.explained_variance_ratio_
                explained_variance = np.sum(pca.explained_variance_ratio_)

                # Log tham số vào MLflow
                mlflow.log_param("algorithm", dim_reduction_method)
                mlflow.log_param("n_components", n_components)
                mlflow.log_param("svd_solver", svd_solver)
                mlflow.log_param("X_train_pca", X_train_pca)
                mlflow.log_metric("explained_variance", explained_variance)

                # Lưu PCA data
                np.save("X_train_pca.npy", X_train_pca)
                mlflow.log_artifact("X_train_pca.npy")

                with col2:
                    st.subheader(
                        f"Hình ảnh kết quả: Giảm xuống còn {n_components} chiều dữ liệu sử dụng phương pháp {dim_reduction_method}")
                    fig2, ax = plt.subplots()
                    scatter = ax.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train[:X_train_pca.shape[0]].astype(
                        int), cmap='tab10', alpha=0.6)
                    legend = ax.legend(
                        *scatter.legend_elements(), title="Digits")
                    ax.add_artist(legend)
                    st.pyplot(fig2)
                    fig2.savefig("pca_result.png")
                    mlflow.log_artifact("pca_result.png")

                st.write("---")
                # Trực quan hóa phương sai giải thích
                st.subheader(
                    "Kết quả trực quan hóa", help="""
- Trong PCA:
    - Tỷ lệ phương sai giải thích là phần trăm phương sai mà mỗi thành phần chính đóng góp vào tổng phương sai 
    của dữ liệu gốc.
        - Ý nghĩa: Tỷ lệ này cho bạn biết mỗi thành phần chính đóng góp bao nhiêu phần trăm vào tổng thông tin của dữ liệu, giúp dễ dàng đánh giá xem 
        bao nhiêu thành phần cần thiết để giữ lại một lượng thông tin nhất định (ví dụ: 90% hoặc 95%). 
""")

            col1, col2 = st.columns([2, 1])
            with col1:
                fig, ax = plt.subplots(1, 1, figsize=(6, 4))

                # # Biểu đồ cột cho explained_variance_
                # ax.bar(range(1, len(pca.explained_variance_) + 1),
                #         pca.explained_variance_)
                # ax.set_title("Phương sai giải thích")
                # ax.set_xlabel("Thành phần chính")
                # ax.set_ylabel("Phương sai")

                # Biểu đồ cột cho explained_variance_ratio_
                ax.bar(
                    range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_)
                ax.set_title("Tỷ lệ phương sai giải thích")
                ax.set_xlabel("Thành phần chính")
                ax.set_ylabel("Tỷ lệ")

                st.pyplot(fig)

            with col2:
                explained_var_r = pca.explained_variance_ratio_
                explanation_data = {
                    "Thành phần chính": [f"Thành phần {i+1}" for i in range(len(explained_var_r))],
                    "Tỷ lệ phương sai giải thích (%)": [f"{var*100:.2f}%" for var in explained_var_r]
                }
                st.table(explanation_data)

            st.success(
                f"Tổng số phương sai giả thích: {sum(pca.explained_variance_ratio_)}")

            st.success(
                f"Log tham số cho **Train_{st.session_state['run_name']}**!")
            st.markdown(
                f"### 🔗 [Truy cập MLflow DAGsHub]({st.session_state['mlflow_url']})")

            mlflow.end_run()

    elif dim_reduction_method == "t-SNE":
        col1, col2 = st.columns(2)
        with col1:
            # Tham số của t-SNE
            n_components = st.selectbox("**Số chiều đầu ra:**", [2, 3],
                                        help="""Số chiều muốn giảm xuống bằng t-SNE. Thường chỉ cần 2 chiều để 
                                        trực quan hóa scatter plot (phù hợp nhất với MNIST). Giá trị 3 có thể 
                                        hữu ích cho phân tích phức tạp hơn, nhưng tăng thời gian tính toán và 
                                        khó trực quan hóa hơn.""")
            perplexity = st.slider(
                "**Perplexity:**", min_value=5, max_value=50, value=30, help="""
Tham số perplexity kiểm soát số lượng điểm lân cận được xem xét trong quá trình giảm chiều. 
Giá trị nhỏ (5-15) phù hợp với dữ liệu nhỏ, giá trị lớn (30-50) phù hợp với dữ liệu lớn hơn. 
Giá trị quá lớn hoặc quá nhỏ có thể làm mất cấu trúc dữ liệu.
""")
            learning_rate = st.slider(
                "**Learning rate:**", min_value=0.001, max_value=0.5, value=0.1, help="""
Tốc độ học (learning rate) điều chỉnh bước di chuyển của thuật toán t-SNE. Giá trị quá nhỏ (0.001-0.01) 
có thể làm chậm quá trình hội tụ, trong khi giá trị quá lớn (0.1, 0.5) có thể gây mất ổn định. Mặc định 
0.01 thường phù hợp với nhiều dữ liệu.
""")
            n_iter = st.slider("**Số vòng lặp tối đa:**",
                               min_value=250, max_value=5000, value=1000, step=250, help="""
Số lần lặp tối đa cho t-SNE để tối ưu hóa. Giá trị nhỏ (250-1000) giảm thời gian tính toán nhưng 
có thể không hội tụ đầy đủ, trong khi giá trị lớn (2000-5000) tăng độ chính xác nhưng kéo dài thời gian. 
Chọn dựa trên kích thước dữ liệu và yêu cầu tốc độ.
""")
            metric = st.selectbox("**Độ đo khoảng cách:**",
                                  ["euclidean", "cosine", "manhattan"], help="""
Độ đo khoảng cách để tính toán sự tương đồng giữa các điểm trong t-SNE:
- `euclidean`: Khoảng cách Euclid (mặc định, phù hợp cho dữ liệu số).
- `cosine`: Khoảng cách Cosine, hữu ích cho dữ liệu vector hướng.
- `manhattan`: Khoảng cách Manhattan, phù hợp với dữ liệu có phân phối đặc biệt.
""")

        if st.button("🚀 Chạy t-SNE"):
            progress_bar = st.progress(0)
            for i in range(1, 101):
                progress_bar.progress(i)
                time.sleep(0.01)
            st.write("Quá trình huấn luyện đã hoàn thành!")
            with mlflow.start_run(run_name=st.session_state["run_name"]) as run:
                mlflow.set_tag("mlflow.runName", st.session_state["run_name"])
                # # Áp dụng t-SNE
                # tsne = TSNE(n_components=n_components, perplexity=perplexity, learning_rate=learning_rate,
                #             n_iter=n_iter, metric=metric, random_state=42)
                X_train_tsne, tsne = fit_tnse(
                    X_train, n_components, perplexity, learning_rate, n_iter, metric)

                # Lưu kết quả t-SNE
                st.session_state.X_train_tsne = X_train_tsne

                try:
                    st.session_state.kl_divergence = tsne.kl_divergence_
                except AttributeError:
                    st.session_state.kl_divergence = "Không có thông tin về k1 divergence"
                mlflow.log_param("algorithm", "t-SNE")
                mlflow.log_param("n_components", n_components)
                mlflow.log_param("perplexity", perplexity)
                mlflow.log_param("learning_rate", learning_rate)
                mlflow.log_param("n_iter", n_iter)
                mlflow.log_param("metric", metric)
                mlflow.log_param("X_train_tsne", X_train_tsne)

                np.save("X_train_tsne.npy", X_train_tsne)
                mlflow.log_artifact("X_train_tsne.npy")

                with col2:
                    st.subheader(
                        f"Hình ảnh kết quả: Giảm xuống còn {n_components} chiều dữ liệu sử dụng phương pháp {dim_reduction_method}")
                    fig2, ax = plt.subplots()
                    scatter = ax.scatter(X_train_tsne[:, 0], X_train_tsne[:, 1], c=y_train[:X_train_tsne.shape[0]].astype(
                        int), cmap='tab10', alpha=0.6)
                    legend = ax.legend(
                        *scatter.legend_elements(), title="Digits")
                    ax.add_artist(legend)
                    st.pyplot(fig2)
                    fig2.savefig("tnse_result.png")
                    mlflow.log_artifact("tnse_result.png")
            st.success(
                f"Log tham số cho **Train_{st.session_state['run_name']}**!")
            st.markdown(
                f"### 🔗 [Truy cập MLflow DAGsHub]({st.session_state['mlflow_url']})")

            mlflow.end_run()
