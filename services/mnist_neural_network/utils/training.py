import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

from sklearn.model_selection import train_test_split,  StratifiedKFold
import os
import mlflow.keras
from tensorflow import keras
import time
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/lbnm203/Machine_Learning_UI.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "lbnm203"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "0902d781e6c2b4adcd3cbf60e0f288a8085c5aab"

    mlflow.set_experiment("MNIST_Neural_Network")


def train_process(X, y):
    mlflow_input()
    st.write("## ⚙️ Quá trình huấn luyện")

    total_samples = X.shape[0]

    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("Chọn số lượng ảnh để train:",
                            1000, total_samples, 10000)
    num_samples = num_samples - 10

    st.session_state.total_samples = num_samples
    # Thanh kéo chọn tỷ lệ Train/Test
    test_size = st.slider("Chọn % dữ liệu Test", 10, 50, 20)
    val_size = st.slider("Chọn % dữ liệu Validation", 0, 50,
                         10)  # Xác định trước khi sử dụng

    remaining_size = 100 - test_size  # Sửa lỗi: Sử dụng test_size thay vì train_size

    # Chọn số lượng ảnh theo yêu cầu
    X_selected, _, y_selected, _ = train_test_split(
        X, y, train_size=num_samples, stratify=y, random_state=42
    )

    # Chia train/test theo tỷ lệ đã chọn
    stratify_option = y_selected if len(np.unique(y_selected)) > 1 else None
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_selected, y_selected, test_size=test_size/100, stratify=stratify_option, random_state=42
    )

    # Chia train/val theo tỷ lệ đã chọn
    stratify_option = y_train_full if len(
        np.unique(y_train_full)) > 1 else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size / remaining_size,
        stratify=stratify_option, random_state=42
    )

    # Lưu vào session_state
    st.session_state.X_train = X_train
    st.session_state.X_val = X_val
    st.session_state.X_test = X_test
    st.session_state.y_train = y_train
    st.session_state.y_val = y_val
    st.session_state.y_test = y_test
    st.session_state.test_size = X_test.shape[0]
    st.session_state.val_size = X_val.shape[0]
    st.session_state.train_size = X_train.shape[0]

    if "X_train" in st.session_state:
        X_train = st.session_state.X_train
        X_val = st.session_state.X_val
        X_test = st.session_state.X_test
        y_train = st.session_state.y_train
        y_val = st.session_state.y_val
        y_test = st.session_state.y_test

    # st.write(f'Training: {X_train.shape[0]} - Testing: {y_test.shape[0]}')

    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    table_size = pd.DataFrame({
        'Dataset': ['Train', 'Validation', 'Test'],
        'Kích thước (%)': [remaining_size - val_size, val_size, test_size],
        'Số lượng mẫu': [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
    })
    st.write(table_size)

    st.write("---")

    X_train, X_val, X_test = [
        st.session_state[k].reshape(-1, 28 * 28) / 255.0 for k in ["X_train", "X_val", "X_test"]]
    y_train, y_val, y_test = [st.session_state[k]
                              for k in ["y_train", "y_val", "y_test"]]

    run_name = st.text_input("Đặt tên Run:", "")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if run_name.strip() == "" or run_name.strip() == " ":
        run_name = f"MNIST_Neural_Network_{timestamp.replace(' ', '_').replace(':', '-')}"
    else:
        run_name = f"{run_name}_{timestamp.replace(' ', '_').replace(':', '-')}"

    st.session_state["run_name"] = run_name

    st.write("---")

    k_folds = st.slider("Số fold cho Cross-Validation:", 3, 10, 5)
    n_hidden = st.slider("Số lớp ẩn:", 1, 5, 2)
    num_neurons = st.slider("Số neuron mỗi lớp:", 32, 512, 128, 32)
    activation = st.selectbox("Hàm kích hoạt:", ["relu", "sigmoid", "tanh"])
    loss_fn = "sparse_categorical_crossentropy"
    epochs = st.slider("Epochs:", 1, 10, 5)

    if st.button("Huấn luyện mô hình"):
        with st.spinner("Đang huấn luyện..."):
            mlflow.start_run(run_name=run_name)

            # Khởi tạo progress bar
            progress_bar = st.progress(0)
            history = None

            # Huấn luyện mô hình với callback để cập nhật progress bar
            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = (epoch + 1) / epochs * 100
                    progress_bar.progress(int(progress))
            mlflow.log_params({"num_layers": n_hidden, "num_neurons": num_neurons,
                              "activation": activation, "optimizer": "adam", "k_folds": k_folds})

            kf = StratifiedKFold(
                n_splits=k_folds, shuffle=True, random_state=42)
            accuracies, losses = [], []

            # for train_idx, val_idx in kf.split(X_train, y_train):
            #     X_k_train, X_k_val = X_train[train_idx], X_train[val_idx]
            #     y_k_train, y_k_val = y_train[train_idx], y_train[val_idx]

            # model = keras.Sequential([layers.Input(shape=(X_k_train.shape[1],))] + [layers.Dense(
            #     num_neurons, activation=activation) for _ in range(n_hidden)] + [layers.Dense(10, activation=activation)])
            num_classes = len(np.unique(y_train))
            # Xây dựng mô hình
            model = models.Sequential([
                layers.Input(shape=(X_train.shape[1],)),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(128, activation='relu'),
                layers.Dense(num_classes, activation='softmax')
            ])
            model.compile(optimizer="adam", loss=loss_fn,
                          metrics=["accuracy"])

            start_time = time.time()
            history = model.fit(X_train, y_train,
                                epochs=epochs,
                                validation_data=(X_val, y_val),
                                verbose=0,
                                callbacks=[ProgressCallback()])

            # Hoàn thành progress bar
            progress_bar.progress(100)
            elapsed_time = time.time() - start_time

            accuracies.append(history.history["val_accuracy"][-1])
            losses.append(history.history["val_loss"][-1])

            avg_val_accuracy = np.mean(accuracies)
            avg_val_loss = np.mean(losses)

            mlflow.log_metrics({"avg_val_accuracy": avg_val_accuracy,
                               "avg_val_loss": avg_val_loss, "elapsed_time": elapsed_time})

            test_loss, test_accuracy = model.evaluate(
                X_test, y_test, verbose=0)
            mlflow.log_metrics(
                {"test_accuracy": test_accuracy, "test_loss": test_loss})
            mlflow.end_run()

            if "models" not in st.session_state:
                st.session_state["models"] = []

            # model_name = "mnist_neural_network"
            count = 1
            new_model_name = run_name
            # Đảm bảo st.session_state["models"] đã được khởi tạo từ trước
            if "models" not in st.session_state:
                st.session_state["models"] = []
            while any(m["name"] == new_model_name for m in st.session_state["models"]):
                new_model_name = f"{run_name}_{count}"
                count += 1

            # Lưu mô hình với tên đã chỉnh sửa
            st.session_state["models"].append(
                {"name": new_model_name, "model": model})

            st.write(f"**Mô hình đã được lưu với tên:** {new_model_name}")

            st.success(f"✅ Huấn luyện hoàn tất!")
            st.success(
                f"**Độ chính xác trung bình trên tập validation:** {avg_val_accuracy:.4f}")
            st.success(f"**Độ chính xác trên tập test:** {test_accuracy:.4f}")
            st.success(
                f"✅ Log dữ liệu **{st.session_state['run_name']}** thành công! 🚀")

            # Đánh giá trên tập test
            st.session_state['trained_model'] = model
            st.session_state['history'] = history

            st.markdown("---")
            with st.spinner("Đang vẽ biểu đồ..."):
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown("#### **Biểu đồ Accuracy và Loss**")
                # Vẽ biểu đồ (xóa các giá trị số)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

                # Biểu đồ Loss
                ax1.plot(history.history['loss'],
                         label='Train Loss', color='blue')
                ax1.plot(history.history['val_loss'],
                         label='Val Loss', color='orange')
                ax1.set_title('Loss')
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss')
                ax1.legend()

                # Biểu đồ Accuracy
                ax2.plot(history.history['accuracy'],
                         label='Train Accuracy', color='blue')
                ax2.plot(history.history['val_accuracy'],
                         label='Val Accuracy', color='orange')
                ax2.set_title('Accuracy')
                ax2.set_xlabel('Epoch')
                ax2.set_ylabel('Accuracy')
                ax2.legend()

                st.pyplot(fig)
                # st.caption("**Biểu đồ Accuracy và Loss**")
