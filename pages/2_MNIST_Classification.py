import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml

from streamlit_drawable_canvas import st_canvas
from services.mnist_classfier.utils.theory import mnist_dataset, decision_tree_theory, full_theory
from services.mnist_classfier.utils.training import train_process
from services.mnist_classfier.utils.demo_st import demo_app
from services.mnist_classfier.utils.show_mlflow import show_experiment_selector


# Streamlit UI
def main():
    st.title("Ứng dụng phân lớp MNIST với Streamlit & MLFlow")

    data_mnist, theory, train, demo, mlflow_p = st.tabs(
        ["Tập dữ liệu", "Thông tin", "Huấn Luyện", "Demo", "MLflow Tracking"])

    # --------------- Data MNIST ---------------
    with data_mnist:
        X, y = mnist_dataset()

    # -------- Theory Decision Tree - SVM ---------
    with theory:
        # decision_tree_theory()
        full_theory()

    # --------------- Training ---------------
    with train:
        train_process(X, y)

    # --------------- DEMO MNIST ---------------
    with demo:
        demo_app()

    # --------------- MLflow Tracking ---------------
    with mlflow_p:
        show_experiment_selector()


if __name__ == "__main__":
    main()
