import streamlit as st

from services.mnist_clustering.utils.data_mnist import mnist_dataset
from services.mnist_clustering.utils.training import train_process
from services.mnist_clustering.utils.theory import explain_kmeans, explain_dbscan
# Streamlit UI


def main():
    st.title(" ✨ MNIST Clustering")

    data_mnist, theory, train, demo = st.tabs(
        ["Tập dữ liệu", "Thông tin", "Huấn Luyện", "Demo"])

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
    with demo:
        pass


if __name__ == "__main__":
    main()
