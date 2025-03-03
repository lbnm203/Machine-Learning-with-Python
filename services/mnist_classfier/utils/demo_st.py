import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import cv2

from PIL import Image
from streamlit_drawable_canvas import st_canvas


def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is not None:
        try:
            img = Image.fromarray(
                canvas_result.image_data[:, :, 0].astype(np.uint8))
            img = img.resize((28, 28)).convert("L")
            img = np.array(img, dtype=np.float32) / 255.0
            return img.reshape(1, -1)
        except Exception as e:
            st.error(f"Lỗi xử lý ảnh: {e}")
            return None
    return None


@st.cache_data
def load_model(path):
    try:
        return joblib.load(path)
    except FileNotFoundError:
        st.error(f"⚠️ Không tìm thấy mô hình tại `{path}`")
        st.stop()


def demo_app():
    st.write("## Demo APP")
    models = {
        "SVM với Linear (mô hình có sẵn)": "./services/mnist_classfier/models/svm_mnist_linear.joblib",
        "SVM với RBF (mô hình có sẵn)": "./services/mnist_classfier/models//svm_mnist_rbf.joblib",
        "SVM với Poly (mô hình có sẵn)": "./services/mnist_classfier/models/svm_mnist_poly.joblib",
        "SVM với Sigmoid (mô hình có sẵn)": "./services/mnist_classfier/models/svm_mnist_sigmoid.joblib",
    }

    # Lấy tên mô hình từ session_state
    model_names = [model["name"]
                   for model in st.session_state.get("models", [])]

    if "models" not in st.session_state:
        st.session_state["models"] = []

    # 📌 Chọn mô hình
    model_option = st.selectbox(
        "🔍 Chọn mô hình:", list(models.keys()) + model_names)

    # Nếu chọn mô hình đã được huấn luyện và lưu trong session_state
    if model_option in model_names:
        model = next(
            model for model in st.session_state["models"] if model["name"] == model_option)["model"]
    else:
        # Nếu chọn mô hình có sẵn (các mô hình đã được huấn luyện và lưu trữ dưới dạng file)
        model = load_model(models[model_option])
        st.success(f"✅ Đã tải mô hình: {model_option}")

    # ✍️ Vẽ số
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            SIZE = 192
            canvas_result = st_canvas(
                fill_color="black",
                stroke_width=10,
                stroke_color="white",
                update_streamlit=True,
                background_color="black",
                height=SIZE,
                width=SIZE,
                drawing_mode="freedraw",
                key="canvas"
            )
            st.write("Canvas data shape:",
                     canvas_result.image_data.shape if canvas_result.image_data is not None else "Không có dữ liệu")

        if st.button("Dự đoán số"):
            with col2:
                img = preprocess_canvas_image(canvas_result)

                if img is not None:
                    # Hiển thị ảnh sau xử lý
                    st.image(Image.fromarray((img.reshape(28, 28) *
                                              255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)

                    # Dự đoán
                    prediction = model.predict(img)
                    st.subheader(f"🔢 Dự đoán: {prediction[0]}")
                    # Tính toán độ tin cậy
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(img)
                        confidence = np.max(proba) * 100
                        st.write(f"Độ tin cậy: {confidence:.2f}%")
                    else:
                        st.warning("Mô hình không hỗ trợ tính toán độ tin cậy.")
                else:
                    st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")
