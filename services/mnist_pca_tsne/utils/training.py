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
import plotly.express as px
import datetime


def input_mlflow():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/lbnm203/Machine_Learning_UI.mlflow/"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "lbnm203"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "0902d781e6c2b4adcd3cbf60e0f288a8085c5aab"
    mlflow.set_experiment("MNIST_PCA_t-SNE")


@st.cache_resource
def fit_tnse(X, n_components):
    tsne = TSNE(n_components=n_components, random_state=42)
    X_train_tsne = tsne.fit_transform(X)
    return X_train_tsne, tsne


def train_pca(X, y):
    total_samples = X.shape[0]

    # Chọn số lượng ảnh để train
    num_samples = st.slider(
        'Chọn số lượng ảnh cho phần huấn luyện', 1000, total_samples, 10000, step=1000)

    X = X.reshape(X.shape[0], -1)
    y = y.reshape(-1)

    # Chọn số lượng ảnh theo yêu cầu
    X_selected, y_selected = X[:num_samples], y[:num_samples]

    st.write(f"- Số lượng mẫu: {X_selected.shape[0]}")

    if "X_selected" not in st.session_state or "y_selected" not in st.session_state:
        st.session_state["X_selected"] = X_selected
        st.session_state["y_selected"] = y_selected

    input_mlflow()

    method = st.selectbox("Chọn phương pháp giảm chiều", ["PCA", "t-SNE"])

    run_name = st.text_input("Đặt tên Run:", "")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if run_name.strip() == "" or run_name.strip() == " ":
        run_name = f"MNIST_{method}_{timestamp.replace(' ', '_').replace(':', '-')}"
    else:
        run_name = f"{run_name}_{method}_{timestamp.replace(' ', '_').replace(':', '-')}"

    st.session_state["run_name"] = run_name

    n_components = st.slider("**Số thành phần chính (n_components):**",
                             min_value=2, max_value=min(X_selected.shape[1], 3),
                             value=3,
                             help="""
Số lượng chiều (`n_components`) muốn giữ lại sau khi giảm chiều bằng PCA.
# """)

    if st.button("Tiến hành giảm chiều"):
        with st.spinner("Đang tiến hành giảm chiều và hiển thị biểu đồ..."):
            mlflow.start_run(run_name=st.session_state["run_name"])
            mlflow.log_param("method", method)
            mlflow.log_param("n_components", n_components)
            mlflow.log_param("num_samples", num_samples)
            mlflow.log_param("original_dim", X.shape[1])

            progress_bar = st.progress(0)

            if method == "t-SNE":
                reducer = TSNE(n_components=n_components, random_state=42)
            else:
                reducer = PCA(n_components=n_components)

            start_time = time.time()
            X_reduced = reducer.fit_transform(X_selected)
            elapsed_time = time.time() - start_time
            mlflow.log_metric("elapsed_time", elapsed_time)

            if method == "PCA":
                explained_variance = np.sum(reducer.explained_variance_ratio_)
                mlflow.log_metric("explained_variance_ratio",
                                  explained_variance)
            elif method == "t-SNE" and hasattr(reducer, "kl_divergence_"):
                mlflow.log_metric("KL_divergence", reducer.kl_divergence_)

            # Update progress bar
            progress_bar.progress(100)

        # Hiển thị kết quả
        if n_components == 2:
            fig = px.scatter(x=X_reduced[:, 0], y=X_reduced[:, 1],
                             color=y_selected.astype(str),
                             title=f"{method} giảm chiều xuống {n_components}D",
                             labels={'x': "Thành phần 1", 'y': "Thành phần 2"})
        else:
            fig = px.scatter_3d(x=X_reduced[:, 0], y=X_reduced[:, 1], z=X_reduced[:, 2],
                                color=y_selected.astype(str),
                                title=f"{method} giảm chiều xuống {n_components}D",
                                labels={'x': "Thành phần 1",
                                        'y': "Thành phần 2", 'z': "Thành phần 3"},)

        st.plotly_chart(fig)

        st.info("""
    "Thành phần 1", "Thành phần 2" và "Thành phần 3" là tên gán cho các trục x, y, z của biểu đồ.

    - Thành phần 1: Đây là trục chứa thông tin có phương sai lớn nhất trong dữ liệu sau khi giảm chiều. 
    Nói cách khác, nó giải thích phần lớn sự biến thiên của dữ liệu.

    - Thành phần 2: Đây là trục chứa thông tin với phương sai thứ hai, giải thích một phần biến thiên còn lại 
    của dữ liệu mà không bị trùng lặp với Thành phần 1.

    - Thành phần 3: Đây là trục chứa thông tin với phương sai tiếp theo, giúp bổ sung thêm thông tin về cấu trúc của dữ liệu.
    """)

        # Lưu kết quả vào MLflow
        os.makedirs("logs", exist_ok=True)
        fig_path = f"logs/{method}_{n_components}D.png"
        fig.write_image(fig_path)
        mlflow.log_artifact(fig_path)

        mlflow.end_run()
        st.success(
            f"✅ Log thành công dữ liệu **Train_{st.session_state['run_name']}**!")
