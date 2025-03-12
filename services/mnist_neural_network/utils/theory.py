import streamlit as st


def neural_network():
    st.write("## Giới thiệu chung")
    st.markdown("Mạng nơ-ron là một phương thức trong lĩnh vực trí tuệ nhân tạo (AI), được sử dụng để dạy máy tính xử lý dữ liệu theo cách "
                "mô phỏng bộ não con người. Đây là một loại quy trình máy học, được gọi là học sâu, sử dụng các nút hoặc nơ-ron liên kết với nhau trong "
                "một cấu trúc phân lớp tương tự như bộ não con người. Phương thức này tạo ra một hệ thống thích ứng được máy tính sử dụng để học hỏi từ "
                "sai lầm của chúng và liên tục cải thiện.")

    st.write("## Kiến trúc")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("./services/mnist_neural_network/assets/nn_3.png")
        #  caption="""
        #  Minh họa mạng nơ ron trên tập dữ liệu MNIST

        #  (http://neuralnetworksanddeeplearning.com)

        #  """)
        st.caption(
            "Minh họa kiến trúc mạng nơ ron"
            "(https://www.v7labs.com/blog/neural-network-architectures-guide)")
    st.markdown("""
Một mạng nơ ron thường bao gồm các lớp:
- **Lớp đầu vào (Input layer)**: Nhận dữ liệu đầu vào. Mỗi nơ-ron đầu vào trong lớp tương ứng với một đặc trưng trong dữ liệu đầu vào.
- **Lớp ẩn (Hidden layers)**: Các lớp này thực hiện hầu hết các nhiệm vụ tính toán phức tạp. Một mạng nơ-ron có thể có một hoặc nhiều lớp ẩn.
Mỗi lớp bao gồm các đơn vị (nơ-ron) chuyển đổi các đầu vào thành thứ mà lớp đầu ra có thể sử dụng.
- **Lớp đầu ra (Output layer)**: Lớp cuối cùng tạo ra đầu ra của mô hình. Định dạng của các đầu ra này thay đổi tùy thuộc vào nhiệm vụ cụ
thể (ví dụ: phân loại, hồi quy).
""")

    st.write("## Cách hoạt động của mạng nơ ron")
    st.markdown("""
### 1. **Feedforward (Lan truyền thẳng)**:
- Khi dữ liệu được đưa vào mạng, nó đi qua mạng theo hướng thuận, từ lớp đầu vào qua các lớp ẩn đến lớp đầu ra.
Quá trình này được gọi là lan truyền thuận. Quá trình này sẽ xảy ra:
    - Biến đổi tuyến tính: Mỗi neuron trong một lớp nhận đầu vào, được nhân với trọng số liên quan đến các kết nối.
    Các tích này được cộng lại với nhau và một độ lệch (bias) được thêm vào tổng""")

    st.latex("""

    z = w_1 x_1 + w_2 x_2 + ... + w_n x_n + b
    """)

    st.markdown(r"""
        $\text{Trong đó:}$

        - $z$ là tổng trọng số đầu vào.

        - $w_i$ là trọng số.

        - $x_i$ là giá trị đầu vào.

        - $b$ là độ lệch (bias).

            - Hàm kích hoạt: Kết quả của phép biến đổi tuyến tính ($z$) được truyền qua một hàm kích hoạt để đưa tính phi tuyến tính vào mạng,
            cho phép mạng học các mẫu phức tạp hơn. Các hàm kích hoạt phổ biến bao gồm ReLU, sigmoid và tanh


""")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("./services/mnist_neural_network/assets/act.png")
        st.caption(
            "Hàm kích hoạt phổ biến trong mạng nơ ron"
            " (https://studymachinelearning.com/activation-functions-in-neural-network)")

    st.write(""" ### 2. **Backpropagation (Lan truyền ngược)**: """)
    st.markdown("""
Sau khi truyền thẳng, mạng đánh giá hiệu suất của nó bằng hàm mất mát, hàm này đo lường sự khác biệt giữa đầu ra thực tế và đầu ra dự đoán.
Mục tiêu của quá trình đào tạo là giảm thiểu tổn thất này. Đây là lúc truyền ngược phát huy tác dụng:

- Tính toán tổn thất (loss): Mạng tính toán tổn thất (loss), hàm này cung cấp thước đo lỗi trong các dự đoán.
Hàm mất mát có thể thay đổi; các lựa chọn phổ biến là lỗi bình phương trung bình cho các tác vụ hồi quy
hoặc tổn thất entropy chéo cho phân loại.
""")

    st.latex("L = - \sum_{i=1}^N y_i \log(\widehat{y}_i)")

    st.markdown(r"""
        $\text{Trong đó:}$

    - $L$ là hàm mất mát.

    - $N$ là số lượng mẫu.

    - $y_i$ là nhãn thực tế.

    - $\widehat{y}_i$ là dự đoán của mô hình.

""")

    st.markdown("""
- Tính toán độ dốc (gradient): Mạng tính toán độ dốc của hàm mất mát đối với từng trọng số (weight) và độ lệch (bias) trong mạng.
Điều này liên quan đến việc áp dụng quy tắc chuỗi của phép tính để tìm ra mức độ mỗi phần của lỗi đầu ra có thể được
quy cho từng trọng số và độ lệch.
""")

    st.markdown("""
- Cập nhật trọng số (weight): Sau khi tính toán độ dốc, trọng số và độ lệch được cập nhật bằng thuật toán tối ưu hóa như
giảm độ dốc ngẫu nhiên (SGD). Các trọng số được điều chỉnh theo hướng ngược lại của độ dốc để giảm thiểu tổn thất.
Kích thước của bước thực hiện trong mỗi lần cập nhật được xác định bởi tốc độ học.
""")
    st.latex("w_i \leftarrow w_i - \eta \\frac{\partial L}{\partial w_i}")
    st.latex("b \leftarrow b - \eta \\frac{\partial L}{\partial b}")

    st.markdown(r"""$\text{Trong đó:}$""")
    st.markdown(
        r"$\frac{\partial L}{\partial w_i}$  là gradient của hàm mất mát theo trọng số $w_i$")

    st.markdown(
        r"$\frac{\partial L}{\partial b}$ là gradient của hàm mất mát theo độ lệch b.")

    st.markdown(r"$\eta$ là tốc độ học.")

    st.write("### 3. Lặp lại")
    st.markdown("""
    Quá trình lan truyền về phía trước, tính toán tổn thất, lan truyền ngược và cập nhật trọng số này được lặp lại trong nhiều lần 
    lặp lại trên tập dữ liệu. Theo thời gian, quá trình lặp lại này làm giảm tổn thất và dự đoán của mạng trở nên chính xác hơn.
""")
