import streamlit as st
from PIL import Image


def pseudo_labeling():
    st.write("## Giới thiệu Pseudo Labelling")
    st.markdown("**Pseudo Labeling** là một kỹ thuật học bán giám sát (semi-supervised learning) được sử dụng phổ biến trong lĩnh "
                "vực Machine Learning, đặc biệt khi dữ liệu có nhãn bị hạn chế. Ý tưởng chính là tận dụng một mô hình học có giám sát "
                "(supervised model) đã được huấn luyện trên một tập dữ liệu nhỏ có nhãn để gán nhãn giả (pseudo labels) cho tập dữ liệu "
                "chưa có nhãn.")

    with st.container(border=True):
        st.markdown("### ⚙️ Cách hoạt động")
        st.markdown("""
        Quy trình Pseudo Labeling thường gồm các bước sau:

        1. Huấn luyện mô hình mạng nơ-ron trên tập dữ liệu có nhãn nhỏ (1%).
                    
        2. Dự đoán nhãn cho tập dữ liệu chưa có nhãn (99%) bằng mô hình đã huấn luyện.
                    
        3. Chọn các mẫu có độ tin cậy cao (ngưỡng), thường dựa vào ngưỡng xác suất (ví dụ: chỉ lấy các dự đoán có xác suất trên 90%).
                    
        4. Gán nhãn giả cho các mẫu đã chọn và thêm chúng vào tập dữ liệu huấn luyện.
                    
        5. Huấn luyện lại mô hình với tập dữ liệu mở rộng, gồm cả dữ liệu có nhãn ban đầu và dữ liệu được gán nhãn giả.
                    
        6. Lặp lại quá trình cho đến khi mô hình đã gán hết nhãn cho mẫu hoặc đạt số vòng lặp tối đa.
        """)
        col1, col2, col3 = st.columns([0.5, 2, 0.5])
        with col2:
            st.image("./services/mnist_nn_label/assets/pseudo.png",
                     caption="Minh họa Pseudo Labelling")
