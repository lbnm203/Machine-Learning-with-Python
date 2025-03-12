import streamlit as st

from services.mnist_clustering.utils.data_mnist import mnist_dataset
# from services.mnist_clustering.utils.training import train_process
from services.mnist_clustering.utils.training_v2 import train_process
from services.mnist_clustering.utils.theory import explain_kmeans, explain_dbscan
from services.mnist_clustering.utils.show_mlflow import show_experiment_selector
# from services.mnist_clustering.utils.demo import demo_app
# Streamlit UI


def main():
    st.title(" ✨ MNIST Clustering")

    data_mnist, theory, train, mlflow_p = st.tabs(
        ["Tập dữ liệu", "Thông tin", "Huấn Luyện", "Mlflow Tracking"])

    # --------------- Data MNIST ---------------
    with data_mnist:
        X, y = mnist_dataset()

    # -------- Theory Decision Tree - SVM ---------
    with theory:
        option = st.selectbox(
            'Chọn phương pháp giải thích:',
            ('KMeans', 'DBSCAN')
        )

        if option == 'KMeans':
            explain_kmeans()
        elif option == 'DBSCAN':
            explain_dbscan()

    # --------------- Training ---------------
    with train:
        train_process(X, y)

    # --------------- DEMO MNIST ---------------
    # with demo:
        # demo_app()

    # --------------- MFlow Tracking ---------------
    with mlflow_p:
        show_experiment_selector()


if __name__ == "__main__":
    main()
