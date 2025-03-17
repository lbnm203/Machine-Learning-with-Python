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
from stqdm import stqdm


def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/lbnm203/Machine_Learning_UI.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "lbnm203"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "0902d781e6c2b4adcd3cbf60e0f288a8085c5aab"

    mlflow.set_experiment("MNIST_Neural_Network_Label")

# Hàm tạo model


def create_model(input_shape, num_classes, hidden1_size, hidden2_size, dropout_rate, activ_hd_1, activ_hd_2, learning_rate):
    model = keras.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(hidden1_size, activation=activ_hd_1),
        layers.Dropout(dropout_rate),
        layers.Dense(hidden2_size, activation=activ_hd_2),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Hàm lấy dữ liệu ban đầu


def get_initial_labeled_data(X, y, percentage):
    num_classes = len(np.unique(y))
    X_labeled = []
    y_labeled = []

    for class_id in range(num_classes):
        class_indices = np.where(y == class_id)[0]
        num_samples = int(len(class_indices) * percentage)
        selected_indices = np.random.choice(
            class_indices, num_samples, replace=False)

        X_labeled.append(X[selected_indices])
        y_labeled.append(y[selected_indices])

    return np.concatenate(X_labeled), np.concatenate(y_labeled)

# Hàm visualize kết quả


def visualize_results(history):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    # Vẽ accuracy
    ax1.plot(history['train_acc'], label='Train Accuracy')
    ax1.plot(history['val_acc'], label='Validation Accuracy')
    ax1.plot(history['test_acc'], label='Test Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # # Vẽ loss
    # ax2.plot(history['train_loss'], label='Train Loss')
    # ax2.plot(history['val_loss'], label='Validation Loss')
    # ax2.plot(history['test_loss'], label='Test Loss')
    # ax2.set_title('Model Loss')
    # ax2.set_xlabel('Iteration')
    # ax2.set_ylabel('Loss')
    # ax2.legend()

    # Vẽ số lượng mẫu được gán nhãn
    ax3.plot(history['labeled_samples'], label='Labeled Samples')
    ax3.set_title('Number of Labeled Samples')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Number of Samples')
    ax3.legend()

    plt.tight_layout()
    return fig

# def run(X, y):
#     mlflow_input()

#     st.write("### Chia dữ liệu")
#     total_samples = X.shape[0]

#     # Thanh kéo chọn số lượng ảnh để train
#     num_samples = st.slider("Chọn số lượng ảnh để train:",
#                             1000, total_samples, 10000)
#     st.session_state.total_samples = num_samples

#     # Thanh kéo chọn tỷ lệ Train/Test/Validation
#     test_size = st.slider("Chọn % dữ liệu Test", 10, 50, 20) / 100
#     val_size = st.slider("Chọn % dữ liệu Validation", 0, 50, 10) / 100
#     train_size = 1.0 - test_size - val_size  # Tỷ lệ còn lại cho train

#     if train_size <= 0:
#         st.error("Tổng tỷ lệ Train + Validation + Test phải nhỏ hơn 100%!")
#         return None, None, None, None

#     # Chọn số lượng ảnh theo yêu cầu
#     X_selected, _, y_selected, _ = train_test_split(
#         X, y, train_size=num_samples/total_samples, stratify=y, random_state=42
#     )

#     # Chia train/test/val theo tỷ lệ đã chọn
#     X_temp, X_test, y_temp, y_test = train_test_split(
#         X_selected, y_selected, test_size=test_size/(1.0), stratify=y_selected, random_state=42
#     )
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_temp, y_temp, test_size=val_size/(1.0 - test_size), stratify=y_temp, random_state=42
#     )

#     # Lưu vào session_state
#     st.session_state.X_train = X_train
#     st.session_state.X_val = X_val
#     st.session_state.X_test = X_test
#     st.session_state.y_train = y_train
#     st.session_state.y_val = y_val
#     st.session_state.y_test = y_test
#     st.session_state.test_size = X_test.shape[0]
#     st.session_state.val_size = X_val.shape[0]
#     st.session_state.train_size = X_train.shape[0]

#     # Chuẩn hóa dữ liệu
#     X_train = X_train.reshape(-1, 28 * 28) / 255.0
#     X_val = X_val.reshape(-1, 28 * 28) / 255.0
#     X_test = X_test.reshape(-1, 28 * 28) / 255.0

#     # Hiển thị bảng kích thước
#     table_size = pd.DataFrame({
#         'Dataset': ['Train', 'Validation', 'Test'],
#         'Kích thước (%)': [train_size*100, val_size*100, test_size*100],
#         'Số lượng mẫu': [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
#     })
#     st.write(table_size)

#     st.write("### Tham số huấn luyện")

#     col1, col2 = st.columns(2)

#     with col1:
#         percentage = st.slider("Tỷ lệ dữ liệu labeled ban đầu (%)",
#                                min_value=0.1, max_value=10.0, value=1.0) / 100
#         threshold = st.slider("Ngưỡng confidence",
#                               min_value=0.5, max_value=0.99, value=0.90, step=0.01)
#         max_iterations = st.number_input("Số vòng lặp tối đa",
#                                          min_value=1, max_value=20, value=5)

#     with col2:
#         hidden1_size = st.number_input("Kích thước lớp ẩn 1",
#                                        min_value=32, max_value=512, value=128, step=32)
#         hidden2_size = st.number_input("Kích thước lớp ẩn 2",
#                                        min_value=32, max_value=512, value=64, step=32)

#         epochs = st.number_input("Số epochs mỗi iteration",
#                                  min_value=1, max_value=50, value=10)

#     dropout_rate = 0.2
#     batch_size = 32

#     # Đặt tên run
#     run_name = st.text_input("Đặt tên Run:", "")
#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     if not run_name.strip():
#         run_name = f"MNIST_Train_Process_{timestamp.replace(' ', '_').replace(':', '-')}"
#     else:
#         run_name = f"{run_name}_{timestamp.replace(' ', '_').replace(':', '-')}"
#     st.session_state["run_name"] = run_name

#     # Nút chạy
#     if st.button("Chạy Pseudo Labeling"):
#         # Lấy dữ liệu labeled ban đầu từ tập train
#         X_labeled, y_labeled = get_initial_labeled_data(
#             X_train, y_train, percentage)
#         X_unlabeled = np.delete(X_train, np.where(
#             np.isin(X_train, X_labeled).all(axis=1))[0], axis=0)

#         # Tạo model
#         input_shape = (28 * 28,)  # Đảm bảo input shape phù hợp sau reshape
#         num_classes = len(np.unique(y))
#         model = create_model(input_shape, num_classes,
#                              hidden1_size, hidden2_size, dropout_rate)

#         # Bắt đầu MLflow run
#         run_name = st.session_state.get(
#             "run_name", f"MNIST_Train_Process_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
#         with mlflow.start_run(run_name=run_name):
#             # Log tham số
#             mlflow.log_param("num_samples", num_samples)
#             mlflow.log_param("train_size", train_size)
#             mlflow.log_param("val_size", val_size)
#             mlflow.log_param("test_size", test_size)
#             mlflow.log_param("percentage_labeled", percentage)
#             mlflow.log_param("threshold", threshold)
#             mlflow.log_param("max_iterations", max_iterations)
#             mlflow.log_param("hidden1_size", hidden1_size)
#             mlflow.log_param("hidden2_size", hidden2_size)
#             mlflow.log_param("dropout_rate", dropout_rate)
#             mlflow.log_param("epochs_per_iteration", epochs)
#             mlflow.log_param("batch_size", batch_size)

#             # Lưu lịch sử
#             history = {'train_acc': [], 'val_acc': [], 'test_acc': [],
#                        'train_loss': [], 'val_loss': [], 'test_loss': [],
#                        'labeled_samples': [len(X_labeled)]}

#             for iteration in stqdm(range(max_iterations), desc="Training Progress"):
#                 st.write(f"\nIteration {iteration + 1}/{max_iterations}")

#                 # Huấn luyện
#                 model.fit(X_labeled, y_labeled,
#                           epochs=epochs,
#                           batch_size=batch_size,
#                           validation_data=(X_val, y_val),
#                           verbose=1)

#                 # Đánh giá
#                 train_loss, train_acc = model.evaluate(
#                     X_labeled, y_labeled, verbose=0)
#                 val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
#                 test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

#                 history['train_acc'].append(train_acc)
#                 history['val_acc'].append(val_acc)
#                 history['test_acc'].append(test_acc)
#                 history['train_loss'].append(train_loss)
#                 history['val_loss'].append(val_loss)
#                 history['test_loss'].append(test_loss)

#                 # Log metrics cho mỗi iteration
#                 mlflow.log_metric("train_accuracy", train_acc, step=iteration)
#                 mlflow.log_metric("val_accuracy", val_acc, step=iteration)
#                 mlflow.log_metric("test_accuracy", test_acc, step=iteration)
#                 mlflow.log_metric("train_loss", train_loss, step=iteration)
#                 mlflow.log_metric("val_loss", val_loss, step=iteration)
#                 mlflow.log_metric("test_loss", test_loss, step=iteration)
#                 mlflow.log_metric("labeled_samples", len(
#                     X_labeled), step=iteration)

#                 st.write(f"Train accuracy: {train_acc:.4f}")
#                 st.write(f"Validation accuracy: {val_acc:.4f}")
#                 st.write(f"Test accuracy: {test_acc:.4f}")

#                 if len(X_unlabeled) == 0:
#                     st.write("Đã gán nhãn hết dữ liệu!")
#                     break

#                 # Dự đoán trên tập unlabeled
#                 predictions = model.predict(X_unlabeled)
#                 confidence_scores = np.max(predictions, axis=1)
#                 pseudo_labels = np.argmax(predictions, axis=1)

#                 # Chọn mẫu vượt ngưỡng
#                 confident_mask = confidence_scores >= threshold
#                 X_confident = X_unlabeled[confident_mask]
#                 y_confident = pseudo_labels[confident_mask]

#                 if len(X_confident) == 0:
#                     st.write("Không còn mẫu nào vượt ngưỡng confidence!")
#                     break

#                 # Cập nhật tập dữ liệu
#                 X_labeled = np.concatenate([X_labeled, X_confident])
#                 y_labeled = np.concatenate([y_labeled, y_confident])
#                 X_unlabeled = X_unlabeled[~confident_mask]

#                 history['labeled_samples'].append(len(X_labeled))

#                 st.write(f"Số mẫu được gán nhãn: {len(X_confident)}")
#                 st.write(f"Số mẫu unlabeled còn lại: {len(X_unlabeled)}")

#             # Log model cuối cùng
#             mlflow.keras.log_model(model, "model")

#             if "models" not in st.session_state:
#                 st.session_state["models"] = []

#             model_name = "mnist_pseudo_label"
#             count = 1
#             new_model_name = model_name
#             while any(m["name"] == new_model_name for m in st.session_state["models"]):
#                 new_model_name = f"{model_name}_{count}"
#                 count += 1

#             st.session_state["models"].append(
#                 {"name": new_model_name, "model": model})
#             st.write(
#                 f"**Mô hình đã được lưu với tên:** `{new_model_name}`")

#             st.success(f"✅ Huấn luyện hoàn tất!")
#             st.write(
#                 f"Train accuracy cuối cùng: {history['train_acc'][-1]:.4f}")
#             st.write(
#                 f"Validation accuracy cuối cùng: {history['val_acc'][-1]:.4f}")
#             st.write(
#                 f"Test accuracy cuối cùng: {history['test_acc'][-1]:.4f}")
#             st.write(
#                 f"Train loss cuối cùng: {history['train_loss'][-1]:.4f}")
#             st.write(
#                 f"Validation loss cuối cùng: {history['val_loss'][-1]:.4f}")
#             st.write(
#                 f"Test loss cuối cùng: {history['test_loss'][-1]:.4f}")
#             st.success(
#                 f"✅ Log dữ liệu **{st.session_state['run_name']}** thành công! 🚀")

#             st.write("---")
#             # Visualize kết quả
#             with st.spinner("Đang trực quan hóa kết quả huấn luyện ..."):
#                 st.subheader("Biểu đồ kết quả")
#                 fig = visualize_results(history)
#                 fig.savefig("results_plot.png")
#                 mlflow.log_artifact("results_plot.png")
#                 st.pyplot(fig)


def run(X, y):
    mlflow_input()

    st.write("### Chia dữ liệu")
    total_samples = X.shape[0]

    # Thanh kéo chọn số lượng ảnh để train
    num_samples = st.slider("Chọn số lượng ảnh để train:",
                            1000, total_samples, 10000)
    num_samples = num_samples - 10
    st.session_state.total_samples = num_samples

    # Thanh kéo chọn tỷ lệ Train/Test/Validation
    test_size = st.slider("Chọn % dữ liệu Test", 10, 50, 20)
    val_size = st.slider("Chọn % dữ liệu Validation", 0, 50, 10)
    train_size = 100 - test_size

    if train_size <= 0:
        st.error("Tổng tỷ lệ Train + Validation + Test phải nhỏ hơn 100%!")
        return None, None, None, None

    # Chọn số lượng ảnh theo yêu cầu

    X_selected, _, y_selected, _ = train_test_split(
        X, y, train_size=num_samples, stratify=y, random_state=42)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_selected, y_selected, test_size=test_size/100, stratify=y_selected, random_state=42)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size / (100 - test_size), stratify=y_train_full, random_state=42)

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

    # Chuẩn hóa dữ liệu
    X_train = X_train.reshape(-1, 28 * 28) / 255.0
    X_val = X_val.reshape(-1, 28 * 28) / 255.0
    X_test = X_test.reshape(-1, 28 * 28) / 255.0

    # Hiển thị bảng kích thước
    table_size = pd.DataFrame({
        'Dataset': ['Train', 'Validation', 'Test'],
        'Kích thước (%)': [train_size*100, val_size*100, test_size*100],
        'Số lượng mẫu': [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
    })
    st.write(table_size)

    st.write("### Tham số huấn luyện")

    col1, col2 = st.columns(2)

    with col1:
        percentage = st.slider("Tỷ lệ dữ liệu labeled ban đầu (%)",
                               min_value=0.1, max_value=10.0, value=0.1) / 100
        threshold = st.slider("Ngưỡng confidence",
                              min_value=0.5, max_value=0.99, value=0.95, step=0.01)
        max_iterations = st.number_input("Số vòng lặp tối đa",
                                         min_value=1, max_value=20, value=5)

        learning_rate = st.number_input("Tốc độ học",
                                        min_value=0.001, max_value=1.0, value=0.01)
        epochs = st.number_input("Số epochs mỗi iteration",
                                 min_value=1, max_value=50, value=10)

    with col2:
        hidden1_size = st.number_input("Kích thước lớp ẩn 1",
                                       min_value=32, max_value=512, value=128, step=32)
        hidden2_size = st.number_input("Kích thước lớp ẩn 2",
                                       min_value=32, max_value=512, value=64, step=32)

        activ_hd_1 = st.selectbox("Hàm kích hoạt lớp ẩn 1:", [
                                  "relu", "sigmoid", "tanh"])

        activ_hd_2 = st.selectbox("Hàm kích hoạt lớp ẩn 2:", [
                                  "relu", "sigmoid", "tanh"])

    dropout_rate = 0.2
    batch_size = 32

    # Đặt tên run
    run_name = st.text_input("Đặt tên Run:", "")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not run_name.strip():
        run_name = f"MNIST_Train_Process_{timestamp.replace(' ', '_').replace(':', '-')}"
    else:
        run_name = f"{run_name}_{timestamp.replace(' ', '_').replace(':', '-')}"
    st.session_state["run_name"] = run_name
    X_labeled, y_labeled = get_initial_labeled_data(
        X_train, y_train, percentage)
    X_unlabeled = np.delete(X_train, np.where(
        np.isin(X_train, X_labeled).all(axis=1))[0], axis=0)

    # Tạo model
    input_shape = (28 * 28,)
    num_classes = len(np.unique(y))
    # Nút chạy
    if st.button("Chạy Pseudo Labeling"):
        with st.spinner("Đang huấn luyện model và log với MLflow..."):
            # Callback để cập nhật thanh trạng thái trong mỗi vòng
            status_text = st.empty()
            progress_bar = st.progress(0)

            class ProgressCallback(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    progress = ((iteration * epochs) + (epoch + 1)
                                ) / (max_iterations * epochs)
                    progress_bar.progress(min(int(progress * 100), 100))
                    status_text.text(
                        f"Tiến trình huấn luyện: {int(progress * 100)}%")
            model = create_model(input_shape, num_classes, hidden1_size, hidden2_size,
                                 dropout_rate, activ_hd_1, activ_hd_2, learning_rate)
            # Bắt đầu MLflow run
            with mlflow.start_run(run_name=run_name):
                # Log tham số
                mlflow.log_param("num_samples", num_samples)
                mlflow.log_param("train_size", train_size)
                mlflow.log_param("val_size", val_size)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("percentage_labeled", percentage)
                mlflow.log_param("threshold", threshold)
                mlflow.log_param("max_iterations", max_iterations)
                mlflow.log_param("hidden1_size", hidden1_size)
                mlflow.log_param("hidden1_method", activ_hd_1)
                mlflow.log_param("hidden2_size", hidden2_size)
                mlflow.log_param("hidden2_method", activ_hd_2)
                mlflow.log_param("dropout_rate", dropout_rate)
                mlflow.log_param("epochs_per_iteration", epochs)
                mlflow.log_param("batch_size", batch_size)
                mlflow.log_param("learning_rate", learning_rate)

            # Lưu lịch sử
            history = {'train_acc': [], 'val_acc': [], 'test_acc': [],
                       'train_loss': [], 'val_loss': [], 'test_loss': [],
                       'labeled_samples': [len(X_labeled)]}

            st.write(
                f"Trước huấn luyện: X_labeled = {len(X_labeled)}, X_unlabeled = {len(X_unlabeled)}")

            # Tạo thanh tiến trình
            progress_bar = st.progress(0)

            for iteration in range(max_iterations):
                # Cập nhật thanh tiến trình
                progress = (iteration + 1) / max_iterations
                progress_bar.progress(progress)

                st.write(f"\nIteration {iteration + 1}/{max_iterations}")

                # Huấn luyện
                model.fit(X_labeled, y_labeled,
                          epochs=epochs,
                          batch_size=batch_size,
                          validation_data=(X_val, y_val),
                          verbose=0,
                          callbacks=[ProgressCallback()])

                # Đánh giá
                train_loss, train_acc = model.evaluate(
                    X_labeled, y_labeled, verbose=0)
                val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
                test_loss, test_acc = model.evaluate(
                    X_test, y_test, verbose=0)

                history['train_acc'].append(train_acc)
                history['val_acc'].append(val_acc)
                history['test_acc'].append(test_acc)

                # Log metrics cho mỗi iteration
                mlflow.log_metric(f"train_accuracy_{iteration+1}",
                                  train_acc, step=iteration)
                mlflow.log_metric("val_accuracy", val_acc, step=iteration)
                mlflow.log_metric(
                    f"test_accuracy_{iteration+1}", test_acc, step=iteration)
                mlflow.log_metric(
                    f"train_loss_{iteration+1}", train_loss, step=iteration)
                mlflow.log_metric(
                    f"val_loss_{iteration+1}", val_loss, step=iteration)
                mlflow.log_metric(
                    f"test_loss_{iteration+1}", test_loss, step=iteration)
                mlflow.log_metric(f"labeled_samples_{iteration+1}", len(
                    X_labeled), step=iteration)

                st.write(f"Train accuracy: {train_acc:.4f}")
                st.write(f"Validation accuracy: {val_acc:.4f}")
                st.write(f"Test accuracy: {test_acc:.4f}")

                if len(X_unlabeled) == 0:
                    st.write("Đã gán nhãn hết dữ liệu!")
                    break

                # Dự đoán trên tập unlabeled
                predictions = model.predict(X_unlabeled)
                confidence_scores = np.max(predictions, axis=1)
                pseudo_labels = np.argmax(predictions, axis=1)

                # Chọn mẫu vượt ngưỡng
                confident_mask = confidence_scores >= threshold
                X_confident = X_unlabeled[confident_mask]
                y_confident = pseudo_labels[confident_mask]

                if len(X_confident) == 0:
                    st.write("Không còn mẫu nào vượt ngưỡng confidence!")
                    break

                # Cập nhật tập dữ liệu
                X_labeled = np.concatenate([X_labeled, X_confident])
                y_labeled = np.concatenate([y_labeled, y_confident])
                X_unlabeled = X_unlabeled[~confident_mask]

                history['labeled_samples'].append(len(X_labeled))

                st.write(f"Số mẫu được gán nhãn: {len(X_confident)}")
                st.write(f"Số mẫu unlabeled còn lại: {len(X_unlabeled)}")

            # Log model cuối cùng
            mlflow.keras.log_model(model, "model")

            if "models" not in st.session_state:
                st.session_state["models"] = []

            model_name = "mnist_pseudo_label"
            count = 1
            new_model_name = model_name
            while any(m["name"] == new_model_name for m in st.session_state["models"]):
                new_model_name = f"{model_name}_{count}"
                count += 1

            st.session_state["models"].append(
                {"name": new_model_name, "model": model})
            st.write(
                f"**Mô hình đã được lưu với tên:** `{new_model_name}`")

            st.success(f"✅ Huấn luyện hoàn tất!")
            st.write(
                f"Train accuracy cuối cùng: {history['train_acc'][-1]:.4f}")
            st.write(
                f"Validation accuracy cuối cùng: {history['val_acc'][-1]:.4f}")
            st.write(
                f"Test accuracy cuối cùng: {history['test_acc'][-1]:.4f}")
            st.write(
                f"Train loss cuối cùng: {history['train_loss'][-1]:.4f}")
            st.write(
                f"Validation loss cuối cùng: {history['val_loss'][-1]:.4f}")
            st.write(
                f"Test loss cuối cùng: {history['test_loss'][-1]:.4f}")
            st.success(
                f"✅ Log dữ liệu **{st.session_state['run_name']}** thành công! 🚀")

        # Visualize kết quả
        with st.spinner("Đang trực quan hóa kết quả huấn luyện ..."):
            st.subheader("Biểu đồ kết quả")
            fig = visualize_results(history)
            fig.savefig("results_plot.png")
            mlflow.log_artifact("results_plot.png")
            st.pyplot(fig)
