import streamlit as st
import numpy as np
import random
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
from sklearn.decomposition import PCA

def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is not None:
        img = Image.fromarray(
            canvas_result.image_data[:, :, 0].astype(np.uint8))
        # Resize và chuyển thành grayscale
        img = img.resize((28, 28)).convert("L")
        img = np.array(img, dtype=np.float32) / 255.0  # Chuẩn hóa về [0, 1]
        return img.reshape(1, -1)  # Chuyển thành vector 1D
    return None


def demo_app():
    st.header("✍️ Vẽ dữ liệu để dự đoán cụm")

    # Kiểm tra danh sách mô hình đã huấn luyện
    if "models" not in st.session_state or not st.session_state["models"]:
        st.warning("⚠️ Không có mô hình nào được lưu! Hãy huấn luyện trước.")
        return

    # Lấy danh sách mô hình đã lưu
    model_names = [model["name"] for model in st.session_state["models"]]

    # 📌 Chọn mô hình
    model_option = st.selectbox("🔍 Chọn mô hình đã huấn luyện:", model_names)
    model = next(m["model"] for m in st.session_state["models"]
                 if m["name"] == model_option)

    # 🆕 Cập nhật key cho canvas khi nhấn "Tải lại"
    if "key_value" not in st.session_state:
        st.session_state.key_value = str(random.randint(0, 1000000))

    if st.button("🔄 Tải lại"):
        st.session_state.key_value = str(random.randint(0, 1000000))
        st.rerun()

    # ✍️ Vẽ dữ liệu
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=10,
        stroke_color="white",
        background_color="black",
        height=150,
        width=150,
        drawing_mode="freedraw",
        key=st.session_state.key_value,
        update_streamlit=True
    )

    if st.button("Dự đoán cụm"):
        img = preprocess_canvas_image(canvas_result)

        if img is not None:
            X_train = st.session_state["X_train"]
            # Hiển thị ảnh sau xử lý
            st.image(Image.fromarray((img.reshape(28, 28) *
                     255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)

            pca = PCA(n_components=2)
            pca.fit(X_train)
            img_reduced = pca.transform(
                img.squeeze().reshape(1, -1))  # Sửa lỗi

            # Dự đoán với K-Means hoặc DBSCAN
            if isinstance(model, KMeans):
                predicted_cluster = model.predict(
                    img_reduced)[0]  # Dự đoán từ ảnh đã PCA
                st.subheader(f"🔢 Cụm dự đoán: {predicted_cluster}")

            elif isinstance(model, DBSCAN):
                model.fit(X_train)  # Fit trước với tập huấn luyện
                predicted_cluster = model.fit_predict(img_reduced)[0]
                if predicted_cluster == -1:
                    st.subheader("⚠️ Điểm này không thuộc cụm nào!")
                else:
                    st.subheader(f"🔢 Cụm dự đoán: {predicted_cluster}")

        else:
            st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")
