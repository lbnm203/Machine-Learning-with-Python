import streamlit as st

import mlflow
import mlflow.sklearn


# from streamlit_drawable_canvas import st_canvas
from services.mnist_nn_label.utils.data_mnist import mnist_dataset
from services.mnist_nn_label.utils.theory import pseudo_labeling
from services.mnist_nn_label.utils.training import run
# from services.mnist_neural_network.utils.demo import demo_app
from services.mnist_nn_label.utils.show_mlflow import show_experiment_selector


# Streamlit UI
def main():
    st.title("MNIST Neural Network Labeling")

    data_mnist, theory, train, mlflow_p = st.tabs(
        ["Tập dữ liệu", "Thông tin", "Huấn Luyện", "MLflow Tracking"])
    # data_mnist, theory, train, demo, mlflow_p = st.tabs(
    #     ["Tập dữ liệu", "Thông tin", "Huấn Luyện", "Demo", "MLflow Tracking"])

    # --------------- Data MNIST ---------------
    with data_mnist:
        X, y = mnist_dataset()

    # -------- Theory Decision Tree - SVM ---------
    with theory:
        pseudo_labeling()
        # pass

    # --------------- Training ---------------
    with train:
        run(X, y)
        # pass

    # --------------- DEMO MNIST ---------------
    # with demo:
    #     # demo_app()
    #     pass

    # --------------- MLflow Tracking ---------------
    with mlflow_p:
        show_experiment_selector()


if __name__ == "__main__":
    main()
