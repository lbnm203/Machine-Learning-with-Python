import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Flatten
import plotly.graph_objects as go

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

# Callback để cập nhật thanh tiến trình


class ProgressCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(
            f"Epoch {epoch + 1}/{self.total_epochs} - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}")

# Hàm xử lý ảnh


def preprocess_image(image):
    image = image.astype('float32') / 255.0
    image = image.reshape(-1, 28, 28)
    return image

# Hàm chọn dữ liệu ban đầu dựa trên số lượng mẫu hoặc tỷ lệ


def select_initial_data(X, y, labeled_ratio=0.01):
    X_selected = []
    y_selected = []
    for digit in range(10):
        indices = np.where(y == digit)[0]
        # Đảm bảo ít nhất 1 mẫu mỗi class
        num_samples = max(1, int(len(indices) * labeled_ratio))
        selected_indices = np.random.choice(
            indices, num_samples, replace=False)
        X_selected.append(X[selected_indices])
        y_selected.append(y[selected_indices])
    return np.concatenate(X_selected), np.concatenate(y_selected)

# Hàm hiển thị ví dụ ảnh được gán nhãn giả


def display_pseudo_labeled_examples(X_unlabeled, y_unlabeled_true, pseudo_labels, confidences, high_confidence_indices, iteration):
    st.subheader(
        f"Ví dụ một số mẫu được gán nhãn giả trong vòng lặp {iteration + 1}")
    num_examples = min(5, len(high_confidence_indices))
    if num_examples == 0:
        st.write(
            "Không có mẫu nào vượt ngưỡng độ tin cậy và được gán đúng trong vòng lặp này.")
        return

    sample_indices = np.random.choice(
        high_confidence_indices, num_examples, replace=False)

    cols = st.columns(num_examples)
    for i, idx in enumerate(sample_indices):
        if idx >= len(X_unlabeled):
            st.warning(
                f"Chỉ số {idx} vượt quá kích thước của X_unlabeled ({len(X_unlabeled)}). Bỏ qua mẫu này.")
            continue

        img = X_unlabeled[idx].reshape(28, 28)
        pred_label = pseudo_labels[idx]
        true_label = y_unlabeled_true[idx]
        conf = confidences[idx]

        with cols[i]:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(
                f"Pred: {pred_label}\nTrue: {true_label}\nConf: {conf:.2f}")
            st.pyplot(fig)

# Hàm huấn luyện với Pseudo Labeling


def train_mnist_pseudo_labeling(X_full, y_full):
    mlflow_input()
    # Khởi tạo session state cho learning rate
    if 'learning_rate' not in st.session_state:
        st.session_state['learning_rate'] = 0.001  # Giá trị mặc định hợp lý

    # Bước 1: Chọn số lượng mẫu và chia tập dữ liệu
    st.subheader("Chia Tập Dữ Liệu")
    num_samples = st.slider("Chọn số lượng ảnh để huấn luyện", min_value=0, max_value=len(X_full), value=min(10000, len(y_full)), step=1000,
                            help=f"Số lượng mẫu để huấn luyện (tối đa {len(X_full)}).")
    if num_samples == 0:
        st.error("Số mẫu phải lớn hơn 0!")
        st.stop()

    # Lấy mẫu từ dữ liệu gốc
    indices = np.random.choice(len(X_full), num_samples, replace=False)
    X_selected, y_selected = X_full[indices], y_full[indices]

    test_ratio = st.slider("Tỷ lệ tập Test (%)", min_value=10.0, max_value=80.0, value=20.0, step=1.0,
                           help="Tỷ lệ dữ liệu dùng để test (10-80%).")
    max_val_ratio = 100.0 - test_ratio
    val_ratio = st.slider("Tỷ lệ Validation (%)", min_value=0.0, max_value=max_val_ratio, value=10.0, step=1.0,
                          help=f"Tỷ lệ dữ liệu dùng để validation (tối đa {max_val_ratio}%).")
    train_ratio = 100.0 - test_ratio - val_ratio

    if train_ratio <= 0:
        st.error("Tỷ lệ Train phải lớn hơn 0! Giảm tỷ lệ Test hoặc Validation.")
        st.stop()

    test_size = test_ratio / 100.0
    val_size = val_ratio / 100.0

    # Chia tập dữ liệu
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_selected, y_selected, test_size=test_size, random_state=42, stratify=y_selected)

    if val_ratio > 0:
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp)
    else:
        X_train, y_train, X_val, y_val = X_temp, y_temp, np.array(
            []), np.array([])

    # Hiển thị bảng dữ liệu
    table_size = pd.DataFrame({
        'Dataset': ['Train', 'Validation', 'Test'],
        'Kích thước (%)': [train_ratio, val_ratio, test_ratio],
        'Số lượng mẫu': [len(X_train), len(X_val), len(X_test)]
    })
    st.write(table_size)

    # Chuẩn hóa dữ liệu
    X_train = preprocess_image(X_train)
    X_val = preprocess_image(X_val)
    X_test = preprocess_image(X_test)
    y_train_cat = to_categorical(y_train, 10)
    y_val_cat = to_categorical(y_val, 10)
    y_test_cat = to_categorical(y_test, 10)

    # Bước 2: Lấy dữ liệu ban đầu (1% làm labeled)
    labeled_ratio = st.slider(
        "% Số mẫu gán nhãn", min_value=0.01, max_value=0.3, value=0.01, step=0.01)
    X_labeled, y_labeled = select_initial_data(X_train, y_train, labeled_ratio)
    y_labeled_cat = to_categorical(y_labeled, 10)
    st.write(
        f"Số mẫu được gán nhãn ban đầu ({labeled_ratio * 100}% mỗi class): {len(X_labeled)}")

    # Tạo tập dữ liệu chưa được gán nhãn (99% còn lại)
    labeled_indices = np.random.choice(
        len(X_train), len(X_labeled), replace=False)
    unlabeled_indices = np.setdiff1d(np.arange(len(X_train)), labeled_indices)
    X_unlabeled = X_train[unlabeled_indices]
    y_unlabeled_true = y_train[unlabeled_indices]
    st.write(f"Số mẫu chưa được gán nhãn: {len(X_unlabeled)}")

    # Bước 3: Thiết lập tham số Neural Network
    st.subheader("Tham Số Neural Network")
    # col1, col2 = st.columns(2)
    # with col1:
    n_hidden_layers = st.slider(
        "Số lớp ẩn", min_value=1, max_value=10, value=2, step=1)
    neurons_per_layer = st.number_input(
        "Số nơ-ron mỗi lớp", min_value=16, max_value=512, value=128, step=16)
    epochs = st.slider("Số vòng lặp (epochs)", min_value=1,
                       max_value=50, value=10, step=1)
    # with col2:
    # batch_size = st.number_input("Kích thước batch", min_value=16, max_value=256, value=32, step=16,
    #                              help="Kích thước batch cho mỗi lần cập nhật trọng số (32-64 là tốt).")
    learning_rate = st.slider("Tốc độ học (η)", min_value=0.001, max_value=0.1, value=float(
        st.session_state['learning_rate']), step=0.001, key="learning_rate_input")
    activation = st.selectbox("Hàm kích hoạt", ['relu', 'sigmoid', 'tanh', 'softmax'], index=0,
                              help="'relu' thường hiệu quả nhất cho MNIST.")

    # Cập nhật learning_rate trong session state
    if learning_rate != st.session_state['learning_rate']:
        st.session_state['learning_rate'] = learning_rate
        # st.write(f"Đã cập nhật tốc độ học: **{learning_rate}**")

    # Bước 4: Thiết lập tham số Pseudo Labeling
    st.subheader("Tham Số Pseudo Labeling")
    max_iterations = st.number_input(
        "Số bước lặp tối đa", min_value=1, max_value=20, value=10, step=1)
    threshold = st.number_input(
        "Ngưỡng tin cậy", min_value=0.5, max_value=1.0, value=0.90, step=0.01)

    # Tùy chỉnh tên run
    run_name = st.text_input("Đặt tên Run:", "")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not run_name.strip():
        run_name = f"MNIST_Train_Process_{timestamp.replace(' ', '_').replace(':', '-')}"
    else:
        run_name = f"{run_name}_{timestamp.replace(' ', '_').replace(':', '-')}"
    st.session_state["run_name"] = run_name

    # Khởi tạo mô hình
    try:
        model = keras.Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        for _ in range(n_hidden_layers):
            model.add(Dense(neurons_per_layer, activation=activation))
        model.add(Dense(10, activation='softmax'))
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])
    except Exception as e:
        st.error(f"Lỗi khi xây dựng mô hình: {str(e)}")
        return

    # Huấn luyện với Pseudo Labeling
    if st.button("Huấn Luyện"):
        mlflow.set_experiment("MNIST_Neural_Network_Label")
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id

            # Vòng lặp Pseudo Labeling
            X_current = X_labeled.copy()
            y_current = y_labeled_cat.copy()
            iteration = 0
            total_iterations = 0
            history_iterations = []

            # Thêm biến theo dõi
            labeled_counts = [len(X_labeled)]  # Số mẫu đã gán nhãn (tích lũy)
            unlabeled_counts = [len(X_unlabeled)]  # Số mẫu chưa gán nhãn
            cumulative_correct_count = 0  # Tổng số nhãn đúng tích lũy trong Pseudo Labeling
            cumulative_correct_counts = [
                cumulative_correct_count]  # Danh sách tổng tích lũy
            confidence_means = []
            correct_ratios = []
            test_accuracies = []
            val_accuracies = []
            additional_epochs = 0  # Khởi tạo mặc định
            final_val_acc = 0.0  # Khởi tạo mặc định
            final_test_acc = 0.0  # Khởi tạo mặc định

            # Tạo container để hiển thị tiến trình
            progress_container = st.empty()

            # Giai đoạn 1: Gán nhãn giả lặp lại
            while iteration < max_iterations:
                with progress_container.container():
                    st.write(
                        f"**Vòng lặp Gán Nhãn {iteration + 1}/{max_iterations}**")
                    st.write(f"Số mẫu huấn luyện hiện tại: {len(X_current)}")
                    st.write(f"Ngưỡng độ tin cậy hiện tại: {threshold:.4f}")

                    # Bước 2: Huấn luyện mô hình trên tập dữ liệu hiện tại
                    progress_callback = ProgressCallback(epochs)
                    history = model.fit(X_current, y_current, epochs=epochs, batch_size=42, verbose=0,
                                        validation_data=(X_val, y_val_cat), callbacks=[progress_callback])
                    history_iterations.append(history.history)

                    # Đánh giá trên tập validation và test
                    _, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
                    _, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
                    val_accuracies.append(val_acc)
                    test_accuracies.append(test_acc)
                    st.write(f"Độ chính xác Validation: {val_acc:.4f}")
                    st.write(f"Độ chính xác Test: {test_acc:.4f}")

                    # Kiểm tra nếu không còn mẫu chưa gán nhãn
                    if len(X_unlabeled) == 0:
                        st.write(
                            "Không còn mẫu chưa gán nhãn để xử lý. Tiếp tục huấn luyện với tập hiện tại.")
                        labeled_counts.append(len(X_current))
                        unlabeled_counts.append(0)
                        cumulative_correct_counts.append(
                            cumulative_correct_count)
                        confidence_means.append(
                            confidence_means[-1] if confidence_means else 0)
                        correct_ratios.append(
                            correct_ratios[-1] if correct_ratios else 0)
                        iteration += 1
                        continue

                    # Bước 3: Dự đoán nhãn cho dữ liệu chưa gán nhãn
                    predictions = model.predict(X_unlabeled)
                    confidences = np.max(predictions, axis=1)
                    pseudo_labels = np.argmax(predictions, axis=1)

                    # Tính độ tin cậy trung bình
                    confidence_mean = np.mean(confidences)
                    confidence_means.append(confidence_mean)
                    st.write(f"Độ tin cậy trung bình: {confidence_mean:.4f}")

                    # Bước 4: Lọc các mẫu gán đúng và vượt ngưỡng
                    correct_and_confident_indices = np.where(
                        (confidences >= threshold) & (pseudo_labels == y_unlabeled_true))[0]

                    # Tính số lượng và tỷ lệ nhãn giả đúng
                    correct_count = len(correct_and_confident_indices) if len(
                        correct_and_confident_indices) > 0 else 0
                    # Tích lũy số nhãn đúng trong Pseudo Labeling
                    cumulative_correct_count += correct_count
                    cumulative_correct_counts.append(cumulative_correct_count)
                    correct_ratio = correct_count / \
                        len(pseudo_labels) if len(pseudo_labels) > 0 else 0
                    correct_ratios.append(correct_ratio)
                    st.write(
                        f"Số lượng mẫu gán đúng trong vòng lặp này: {correct_count}")
                    st.write(
                        f"Tổng số mẫu gán đúng tích lũy (Pseudo Labeling): {cumulative_correct_count}")
                    st.write(
                        f"Tỷ lệ nhãn giả đúng trong vòng lặp này: {correct_ratio:.4f}")

                    # Cảnh báo nếu tỷ lệ nhãn giả đúng thấp
                    if correct_ratio < 0.7 and iteration > 0:
                        st.warning(
                            "Tỷ lệ nhãn giả đúng quá thấp (< 70%). Hãy xem xét giảm learning rate hoặc tăng ngưỡng độ tin cậy.")

                    # Kiểm tra nếu không có mẫu nào vượt ngưỡng và được gán đúng
                    if len(correct_and_confident_indices) == 0:
                        st.warning(
                            "Không có mẫu nào vượt ngưỡng và được gán đúng trong vòng lặp này. Tiếp tục với dữ liệu hiện tại.")
                        labeled_counts.append(len(X_current))
                        unlabeled_counts.append(len(X_unlabeled))
                        iteration += 1
                        continue

                    X_pseudo = X_unlabeled[correct_and_confident_indices]
                    y_pseudo = pseudo_labels[correct_and_confident_indices]
                    y_pseudo_cat = to_categorical(y_pseudo, 10)

                    # Bước 5: Cập nhật tập dữ liệu huấn luyện
                    X_current = np.concatenate([X_current, X_pseudo])
                    y_current = np.concatenate([y_current, y_pseudo_cat])

                    # Loại bỏ các mẫu đã được gán nhãn khỏi tập chưa gán nhãn
                    remaining_indices = np.setdiff1d(
                        np.arange(len(X_unlabeled)), correct_and_confident_indices)
                    X_unlabeled = X_unlabeled[remaining_indices] if len(
                        remaining_indices) > 0 else np.array([])
                    y_unlabeled_true = y_unlabeled_true[remaining_indices] if len(
                        remaining_indices) > 0 else np.array([])

                    # Hiển thị ví dụ
                    display_pseudo_labeled_examples(
                        X_unlabeled, y_unlabeled_true, pseudo_labels, confidences, correct_and_confident_indices, iteration)

                    # Cập nhật số lượng
                    labeled_counts.append(len(X_current))
                    unlabeled_counts.append(
                        len(X_unlabeled) if len(X_unlabeled) > 0 else 0)
                    iteration += 1

            # Kiểm tra chính xác cuối cùng trên toàn bộ tập train
            final_predictions = model.predict(X_train)
            final_pseudo_labels = np.argmax(final_predictions, axis=1)
            final_correct_count = np.sum(final_pseudo_labels == y_train)
            final_accuracy = final_correct_count / len(y_train)
            st.success(f"Đã hoàn thành gán nhãn sau {iteration} vòng lặp.")
            st.write(
                f"Tổng số mẫu gán đúng trên toàn bộ tập train: {final_correct_count} / {len(y_train)}")
            st.write(f"Tỷ lệ gán đúng cuối cùng: {final_accuracy:.4f}")

            # Cập nhật final_correct_counts để hiển thị final_correct_count ở vòng lặp cuối
            final_correct_counts = [
                0] * (len(cumulative_correct_counts) - 1) + [final_correct_count]

            # Giai đoạn 2: Huấn luyện bổ sung (nếu đạt 100% chính xác)
            if final_accuracy == 1.0:
                st.subheader("6. Huấn Luyện Bổ Sung với Toàn Bộ Dữ Liệu")
                additional_epochs = st.number_input("Số epochs bổ sung", min_value=1, max_value=50, value=5, step=1,
                                                    help="Số epochs để huấn luyện thêm sau khi đạt 100% chính xác.")
                if additional_epochs > 0:
                    progress_callback = ProgressCallback(additional_epochs)
                    history = model.fit(X_train, y_train_cat, epochs=additional_epochs, batch_size=42, verbose=0,
                                        validation_data=(X_val, y_val_cat), callbacks=[progress_callback])
                    history_iterations.append(history.history)

                    # Đánh giá cuối cùng
                    _, final_val_acc = model.evaluate(
                        X_val, y_val_cat, verbose=0)
                    _, final_test_acc = model.evaluate(
                        X_test, y_test_cat, verbose=0)
                    val_accuracies.append(final_val_acc)
                    test_accuracies.append(final_test_acc)
                    st.success(
                        f"Độ chính xác cuối cùng - Validation: {final_val_acc:.4f}, Test: {final_test_acc:.4f}")

                    # Lưu mô hình
                    mlflow.keras.log_model(model, "final_model")

            total_iterations = iteration + (additional_epochs > 0 and 1 or 0)
            st.write(f"Tổng số vòng lặp: {total_iterations}")

            # Biểu đồ trực quan
            st.subheader("7. Biểu Đồ Quá Trình Huấn Luyện Pseudo Labeling")

            # Biểu đồ số lượng mẫu
            fig_counts = go.Figure()
            fig_counts.add_trace(go.Scatter(x=list(range(len(labeled_counts))),
                                 y=labeled_counts, mode='lines+markers', name='Số mẫu đã gán nhãn'))
            fig_counts.add_trace(go.Scatter(x=list(range(len(unlabeled_counts))),
                                 y=unlabeled_counts, mode='lines+markers', name='Số mẫu chưa gán nhãn'))
            fig_counts.update_layout(title="Số lượng mẫu qua các vòng lặp",
                                     xaxis_title="Vòng lặp", yaxis_title="Số lượng", height=400)
            st.plotly_chart(fig_counts, use_container_width=True)

            # Biểu đồ độ chính xác
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=list(range(len(val_accuracies))),
                              y=val_accuracies, mode='lines+markers', name='Validation Accuracy'))
            fig_acc.add_trace(go.Scatter(x=list(range(len(test_accuracies))),
                              y=test_accuracies, mode='lines+markers', name='Test Accuracy'))
            fig_acc.update_layout(title="Độ chính xác qua các vòng lặp",
                                  xaxis_title="Vòng lặp", yaxis_title="Độ chính xác", height=400)
            st.plotly_chart(fig_acc, use_container_width=True)

            # Biểu đồ số lượng mẫu gán đúng (tích lũy)
            fig_correct = go.Figure()
            fig_correct.add_trace(go.Scatter(x=list(range(len(cumulative_correct_counts))),
                                  y=cumulative_correct_counts, mode='lines+markers', name='Tích lũy trong Pseudo Labeling'))
            fig_correct.add_trace(go.Scatter(x=list(range(len(final_correct_counts))), y=final_correct_counts,
                                  mode='lines+markers', name='Trên toàn bộ tập train', line=dict(dash='dash')))
            fig_correct.update_layout(
                title="Tổng số lượng mẫu gán đúng qua các vòng lặp",
                xaxis_title="Vòng lặp",
                yaxis_title="Số lượng",
                # Tự động điều chỉnh dựa trên len(X_train)
                yaxis_range=[0, len(X_train)],
                height=400
            )
            st.plotly_chart(fig_correct, use_container_width=True)

            # Log MLflow
            mlflow.log_params({
                "num_samples": num_samples,
                "test_ratio": test_ratio,
                "val_ratio": val_ratio,
                "train_ratio": train_ratio,
                "labeled_ratio": labeled_ratio,
                "n_hidden_layers": n_hidden_layers,
                "neurons_per_layer": neurons_per_layer,
                "epochs": epochs,
                "batch_size": 42,
                "learning_rate": learning_rate,
                "activation": activation,
                "max_iterations": max_iterations,
                "threshold": threshold,
                "labeling_iterations": iteration,
                "total_iterations": total_iterations,
                "log_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            mlflow.log_metrics({
                "final_val_accuracy": float(final_val_acc),
                "final_test_accuracy": float(final_test_acc),
                "final_train_accuracy": float(final_accuracy),
                "total_correct_pseudo_labels": float(cumulative_correct_count),
                "final_correct_count": float(final_correct_count)
            })

            # ---------------- Thêm đoạn code lưu mô hình ----------------
            # model_name = st.text_input(
            #     "Nhập tên mô hình:", "mnist_pseudo_label")

            # Kiểm tra và tạo tên không trùng
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
            # -------------------------------------------------------------
