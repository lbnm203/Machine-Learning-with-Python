import streamlit as st
from PIL import Image


def theory_linear():
    st.markdown("## Simple Linear Regression")
    st.image("./services/Linear_Regression/assest/simple_lr.png")

    st.write("""
        Hồi quy tuyến tính đơn giản là hồi quy tuyến tính với một biến độc lập, còn được gọi là biến giải thích,
        và một biến phụ thuộc, còn được gọi là biến đáp ứng. Trong hồi quy tuyến tính đơn giản, biến phụ thuộc
        là liên tục. Hồi quy tuyến tính đơn giản giúp đưa ra dự đoán và hiểu được mối quan hệ giữa một biến độc
        lập và một biến phụ thuộc.

    """)

    st.markdown("### Công thức")
    st.markdown("""
        Hồi quy tuyến tính đơn được biểu diễn dưới dạng 1 đường thẳng:
    """)

    st.latex(r"""
        y = w_0 + w_1 x + \epsilon
    """)
    st.markdown("""
        - y là biến phụ thuộc
        - $w_0$ là bias (intercept)
        - $w_1$ là hằng số (slope)
        - x là biến độc lập

    """)

    st.markdown("""## Multiple Linear Regression""")
    st.image("./services/Linear_Regression/assest/mutiple_lr.png")

    st.markdown(
        """Hồi quy tuyến tính bội tương tự như mô hình hồi quy tuyến tính đơn, chúng dùng để dự đoán một giá trị
        của biến phụ thuộc y dựa vào 2 hoặc nhiều biến độc lập x1, x2, x3…xn.""")

    st.markdown("""### Công thức""")
    st.latex(r"""
        y = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n
    """)

    st.markdown(r"""
        Trong đó:
        - $y$: Biến phụ thuộc

        - $w_0$: Bias (intercept)

        - $w_1$, $w_2$, $w_3$,... $w_n$: Hng số (slope)

        - $x_1$, $x_2$, $x_3$,... $x_n$: Biến độc lập
    """)

    st.markdown("""## Polynomial Regression:""")

    st.image("./services/Linear_Regression/assest/poly_lr.png")

    st.markdown("""
    Polynomial Regression là thuật toán hồi quy đa thức, nó giống như thuật toán hồi quy tuyến tính, sử dụng mối quan hệ
    giữa các biến độc lập x và biến phụ thuộc y được biểu diễn dưới dạng đa thức bậc n, để tìm cách tốt nhất
    vẽ một đường qua các điểm dữ liệu sao cho tối ưu và phù hợp nhất.

    """)

    st.markdown("""### Công thức""")
    st.latex(r"""
        y = w_0 + w_1x + w_2x^2 + w_3x^3 + \dots + w_nx^n

    """)

    st.markdown(r"""
        Trong đó:
        - $y$: Biến phụ thuộc

        - $w_0$: Bias (intercept)

        - $w_1$, $w_2$, $w_3$,... $w_n$: Hng số (slope)

        - $x_1$, $x_2$, $x_3$,... $x_n$: Biến độc lập

        - $d$: bậc của đa thức
    """)

    st.markdown("""### Hàm Mất Mát (Loss Function)""")
    st.latex(
        r""" MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y_i})^2""")

    st.markdown(r"""
        Trong đó:
        - $n$: Số lượng điểm dữ liệu.
        - $y_i$: Giá trị thực tế của biến phụ thuộc.
        - $\hat{y}_i$: Giá trị dự đoán từ mô hình.
    """)

    st.markdown("""### Gradient Descent""")

    st.markdown(r"""
    Các bước trong Gradient Descent

- Khởi tạo các trọng số $w_0$, $w_1$, $w_2$, ..., $w_n$ với giá trị bất kỳ.

- Tính gradient của MSE đối với từng trọng số.

- Cập nhật $w_0$, $w_1$, $w_2$ theo quy tắc của thuật toán Gradient Descent.

    """)
    st.latex(
        r"""w_i = w_i - \eta \frac{\partial J}{\partial w_0} """)

    st.markdown(r"""
- Lặp lại bước 2 và 3 cho đến khi gradient gần bằng 0 hoặc không
thay đổi đáng kể nữa.

    """)

    st.markdown("### Đánh giá mô hình:")
    st.markdown("""
                
- **Hệ số tương quan (R)**: R là hệ số tương quan giữa giá trị thực tế của biến phụ thuộc và
    giá trị dự đoán của mô hình. Nó cho biết mức độ tương quan tuyến tính giữa hai tập giá trị này.

- **Hệ số xác định (R²)**: R2 cho biết tỷ lệ phần trăm biến động của biến phụ thuộc có thể
giải thích được bởi các biến độc lập trong mô hình.:
    """)
    st.latex(r"""
    R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
    """)

    st.latex(r"""
    R^2_{adjusted} = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - k - 1} \right)
    """)

    st.markdown(r"""
    Trong đó:
    - $n$: Số lượng quan sát.
    - $k$: Số lượng biến độc lập.
    - $\bar{y}$: Giá trị trung bình của biến phụ thuộc.
    """)
