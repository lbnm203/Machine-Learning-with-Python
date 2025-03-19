import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps


def preprocess_canvas_image(canvas_result):
    if canvas_result.image_data is not None:
        img = Image.fromarray(
            canvas_result.image_data[:, :, 0].astype(np.uint8))
        # Resize và chuyển thành grayscale
        img = img.resize((28, 28)).convert("L")
        img = np.array(img, dtype=np.float32) / 255.0  # Chuẩn hóa về [0, 1]
        return img.reshape(1, 28, 28)  # Đảm bảo shape đúng cho mô hình
    return None


def demo_app():
    st.header("👉 DEMO dự đoán")

    if "models" not in st.session_state or not st.session_state["models"]:
        st.warning("⚠️ Không có mô hình nào được lưu! Hãy huấn luyện trước.")
        return

    # Lấy danh sách mô hình đã lưu
    model_names = [model["name"] for model in st.session_state["models"]]

    # 📌 Chọn mô hình
    model_option = st.selectbox(" Chọn mô hình để dự đoán:", model_names)
    model = next(m["model"] for m in st.session_state["models"]
                 if m["name"] == model_option)

    # 🆕 Cập nhật key cho canvas khi nhấn "Tải lại"
    if "key_value" not in st.session_state:
        st.session_state.key_value = str(random.randint(0, 1000000))

    with st.expander("DEMO"):
        st.write("---")
        col1, col2, col3, col4 = st.columns([1, 3, 3, 1])
        with col1:
            if st.button("🎨 Vẽ Ảnh"):
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

        if st.button("Dự đoán số"):
            with col2:
                img = preprocess_canvas_image(canvas_result)

            if img is not None:
                st.image(Image.fromarray(
                    (img.reshape(28, 28) * 255).astype(np.uint8)), caption="Ảnh sau xử lý", width=100)

                # Dự đoán số
                prediction = model.predict(img)
                predicted_number = np.argmax(prediction, axis=1)[0]
                max_confidence = np.max(prediction)

                st.subheader(f"🔢 Dự đoán: {predicted_number}")
                st.write(f"📊 Mức độ tin cậy: {max_confidence:.2%}")

                # Hiển thị bảng confidence score
                prob_df = pd.DataFrame(prediction.reshape(
                    1, -1), columns=[str(i) for i in range(10)]).T
                prob_df.columns = ["Mức độ tin cậy"]
                st.bar_chart(prob_df)

            else:
                st.error("⚠️ Hãy vẽ một số trước khi bấm Dự đoán!")
