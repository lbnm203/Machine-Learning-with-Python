import streamlit as st

from services.mnist_pca_tsne.utils.data_mnist import mnist_dataset
# from services.mnist_clustering.utils.training import train_process
from services.mnist_pca_tsne.utils.theory import introduce_pca, introduce_tsne
# from services.mnist_clustering.utils.show_mlflow import show_experiment_selector
# Streamlit UI


def main():
    st.title(" ✨ Giảm Chiều Dữ Liệu MNIST PCA - t-SNE")

    data_mnist, theory, train, mlflow_p = st.tabs(
        ["Tập dữ liệu", "Thông tin", "Huấn Luyện", "Mlflow Tracking"])

    # --------------- Data MNIST ---------------
    with data_mnist:
        X, y = mnist_dataset()

    # -------- Theory Decision Tree - SVM ---------
    with theory:
        option = st.selectbox(
            'Chọn phương pháp giải thích:',
            ('Thuật toán PCA', 'Thuật toán t-SNE')
        )

        if option == 'Thuật toán PCA':
            introduce_pca()
        elif option == 'Thuật toán t-SNE':
            introduce_tsne()

    # --------------- Training ---------------
    with train:
        pass
        # train_process(X, y)

    # --------------- DEMO MNIST ---------------
    with mlflow_p:
        pass
        # show_experiment_selector()


if __name__ == "__main__":
    main()
