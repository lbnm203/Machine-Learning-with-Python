import streamlit as st


def introduce_pca():
    # Tiêu đề chính
    st.title("Thuật toán PCA ")

    # Phần 1: Giới thiệu chung về PCA
    st.header("1. Giới thiệu chung về PCA (Principal Component Analysis)")
    st.write("""
    PCA (Phân tích Thành phần Chính) là một kỹ thuật giảm chiều dữ liệu, giúp biến đổi dữ liệu từ không gian nhiều chiều sang không gian có ít chiều hơn mà vẫn giữ được phần lớn thông tin quan trọng. 
    PCA thường được sử dụng trong các lĩnh vực như xử lý dữ liệu, học máy và trực quan hóa dữ liệu nhằm:
    - **Giảm chiều dữ liệu**: Giúp cải thiện hiệu suất của các thuật toán học máy và tránh hiện tượng overfitting.
    - **Trực quan hóa**: Biến dữ liệu đa chiều thành dữ liệu 2D hoặc 3D để dễ dàng quan sát.
    - **Loại bỏ nhiễu**: Giảm các thành phần không quan trọng trong dữ liệu.
    """)

    # Phần 2: Ý tưởng chính của PCA
    st.header("2. Ý tưởng chính của PCA")
    st.write("""
    PCA hoạt động dựa trên việc tìm ra các **thành phần chính** (principal components), là các hướng trong không gian dữ liệu mà dữ liệu biến thiên nhiều nhất. Các thành phần chính này được xác định thông qua:
    - **Vectơ riêng** (eigenvectors) của ma trận hiệp phương sai của dữ liệu.
    - **Giá trị riêng** (eigenvalues) tương ứng, thể hiện mức độ biến thiên của dữ liệu theo hướng của vectơ riêng đó.
    Các vectơ riêng được sắp xếp theo thứ tự giảm dần của giá trị riêng, và chúng ta chọn một số lượng thành phần chính (k) để giảm chiều dữ liệu.
    """)

    # Phần 3: Các bước thực hiện PCA
    st.header("3. Các bước thực hiện PCA")
    st.write("""
    Để áp dụng PCA, chúng ta thực hiện theo các bước sau:
    1. **Chuẩn hóa dữ liệu** (nếu cần): Đảm bảo dữ liệu có giá trị trung bình bằng 0 và độ lệch chuẩn bằng 1 (đặc biệt khi các biến có đơn vị khác nhau).
    2. **Tính ma trận hiệp phương sai**: Xác định mức độ tương quan giữa các biến trong dữ liệu.
    3. **Tính vectơ riêng và giá trị riêng**: Từ ma trận hiệp phương sai, tìm các hướng biến thiên chính.
    4. **Sắp xếp vectơ riêng**: Theo thứ tự giảm dần của giá trị riêng.
    5. **Chọn k vectơ riêng**: Lấy k vectơ có giá trị riêng lớn nhất để tạo ma trận chuyển đổi.
    6. **Chiếu dữ liệu**: Biến đổi dữ liệu gốc sang không gian mới bằng ma trận chuyển đổi.
    """)

    # Phần 4: Công thức của PCA
    st.header("4. Công thức toán học của PCA")
    st.write("Dưới đây là các công thức chi tiết của PCA:")
    st.latex(r"""
    \text{Giả sử dữ liệu đầu vào là ma trận } X \text{ với kích thước } n \times p, 
    \text{ trong đó } n \text{ là số mẫu và } p \text{ là số biến.}
    """)
    st.markdown(r"- **Ma trận hiệp phương sai** $\Sigma$ được tính bằng:")
    st.latex(r"""
    \Sigma = \frac{1}{n-1} (X - \bar{X})^T (X - \bar{X})
    """)
    st.markdown(
        r"Trong đó $\bar{X}$ là ma trận chứa giá trị trung bình của các biến.")
    st.markdown(
        r"- **Vectơ riêng** $v_i$ và **giá trị riêng** $\lambda_i$ của $\Sigma$ thỏa mãn:")
    st.latex(r"""
    \Sigma v_i = \lambda_i v_i
    """)
    st.markdown(
        r"- **Ma trận chuyển đổi** $W$ được tạo từ $k$ vectơ riêng có giá trị riêng lớn nhất:")
    st.latex(r"""
    W = [v_1, v_2, \dots, v_k]
    """)
    st.markdown(r"- **Dữ liệu sau khi giảm chiều** $Y$ được tính bằng:")
    st.latex(r"""
    Y = X W
    """)

    # Phần 5: Ứng dụng của PCA
    st.header("5. Ứng dụng của PCA")
    st.write("""
    PCA có nhiều ứng dụng thực tế, bao gồm:
    - **Giảm chiều dữ liệu để trực quan hóa**: Ví dụ, biến dữ liệu đa chiều thành 2D hoặc 3D để vẽ biểu đồ.
    - **Tăng hiệu suất học máy**: Giảm số chiều để giảm thời gian tính toán và nguy cơ overfitting.
    - **Loại bỏ nhiễu**: Tập trung vào các thành phần chính và bỏ qua các thành phần ít quan trọng.
    """)

    # Phần 6: Lưu ý khi sử dụng PCA
    st.header("6. Lưu ý khi sử dụng PCA")
    st.write("""
    Khi sử dụng PCA, cần lưu ý:
    - PCA giả định dữ liệu có **phân phối Gauss** và các thành phần chính là **tuyến tính**.
    - PCA có thể không hiệu quả với dữ liệu có cấu trúc **phi tuyến**.
    - Việc chuẩn hóa dữ liệu là rất quan trọng nếu các biến có thang đo khác nhau.
    """)


def introduce_tsne():
    # Tiêu đề chính
    st.title(
        "Thuật toán t-SNE (t-Distributed Stochastic Neighbor Embedding)")

    # Phần 1: Giới thiệu chung về t-SNE
    st.header("1. Giới thiệu chung về t-SNE")
    st.write("""
    t-SNE là một kỹ thuật **giảm chiều dữ liệu phi tuyến**, được thiết kế để trực quan hóa dữ liệu đa chiều trong không gian 2D hoặc 3D. 
    Thuật toán này rất hữu ích trong việc khám phá các cấu trúc ẩn trong dữ liệu, chẳng hạn như các cụm hoặc nhóm tương tự.  
    - **Mục đích chính**: Biến dữ liệu phức tạp thành biểu diễn dễ quan sát mà vẫn giữ được các mối quan hệ quan trọng giữa các điểm dữ liệu.  
    - **Ưu điểm nổi bật**: Tập trung vào việc bảo toàn cấu trúc cục bộ của dữ liệu, đảm bảo các điểm gần nhau trong không gian gốc vẫn gần nhau trong không gian giảm chiều.  
    - **Người phát triển**: Thuật toán được giới thiệu bởi Laurens van der Maaten và Geoffrey Hinton vào năm 2008.
    """)

    # Phần 2: Ý tưởng chính của t-SNE
    st.header("2. Ý tưởng chính của t-SNE")
    st.write("""
    t-SNE hoạt động dựa trên ba ý tưởng cốt lõi:  
    1. **Đo lường sự tương đồng trong không gian gốc**: Sử dụng phân phối Gaussian để tính xác suất hai điểm là "hàng xóm" của nhau dựa trên khoảng cách giữa chúng.  
    2. **Đo lường sự tương đồng trong không gian giảm chiều**: Sử dụng phân phối t-Student với đuôi dài (heavy-tailed) để mô phỏng sự tương đồng trong không gian mới.  
    3. **Tối ưu hóa**: Điều chỉnh vị trí các điểm trong không gian giảm chiều để giảm thiểu sự khác biệt giữa hai phân phối này, sử dụng hàm mất mát **Kullback-Leibler divergence (KL divergence)**.
    """)

    # Phần 3: Các bước thực hiện t-SNE
    st.header("3. Các bước thực hiện t-SNE")
    st.markdown(r"""
    Quy trình của t-SNE có thể được chia thành các bước sau:  
    1. **Tính ma trận tương đồng trong không gian gốc**:  
       - Với mỗi cặp điểm $i, j$, tính xác suất $p_{ij}$ rằng $i$ và $j$ là hàng xóm, dựa trên phân phối Gaussian.  
       - Tham số $\sigma_i$ (độ lệch chuẩn) được điều chỉnh dựa trên tham số **perplexity** do người dùng chọn.  
    2. **Khởi tạo ngẫu nhiên các điểm trong không gian giảm chiều** (thường là 2D hoặc 3D).  
    3. **Tính ma trận tương đồng trong không gian giảm chiều**:  
       - Sử dụng phân phối t-Student để tính xác suất $q_{ij}$ giữa các điểm trong không gian mới.  
    4. **Tối ưu hóa vị trí các điểm**:  
       - Sử dụng thuật toán **gradient descent** để giảm thiểu hàm mất mát KL divergence giữa $p_{ij}$ và $q_{ij}$.
    """)

    # Phần 4: Công thức toán học của t-SNE
    st.header("4. Công thức toán học của t-SNE")
    st.write("Dưới đây là các công thức cơ bản của t-SNE:")
    st.write(
        "- **Xác suất tương đồng trong không gian gốc** (dựa trên phân phối Gaussian):")
    st.latex(r"""
    p_{ij} = \frac{\exp(\frac{-\|x_i - x_j\|^2}{2\sigma_i^2})}{\sum_{k \neq i} \exp(\frac{-\|x_i - x_k\|^2}{2\sigma_i^2})}
    """)
    st.markdown(r"""
    Trong đó:  
    - $x_i, x_j$: Các điểm dữ liệu trong không gian gốc.  
    - $\sigma_i$: Độ lệch chuẩn của Gaussian  
    """)
    # st.markdown("Sau đó, để đảm bảo đối xứng, ta tính: ")
    # st.latex(r"p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}")
    st.write(
        "- **Xác suất tương đồng trong không gian giảm chiều** (dựa trên phân phối t-Student):")
    st.latex(r"""
    q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
    """)
    st.markdown(r"""
    Trong đó:  
    - $y_i, y_j$: Các điểm dữ liệu trong không gian giảm chiều.  
    - Phân phối t-Student có độ tự do là 1, giúp mô phỏng tốt hơn các khoảng cách trong không gian thấp chiều.  
    """)
    st.write("- **Hàm mất mát KL divergence**:")
    st.latex(r"""
    C = \sum_i \sum_j p_{ij} \log \left( \frac{p_{ij}}{q_{ij}} \right)
    """)
    st.markdown(
        r"Mục tiêu là tối ưu hóa $C$ để $q_{ij}$ gần với $p_{ij}$ nhất có thể.")

    # Phần 5: Ứng dụng của t-SNE
    st.header("5. Ứng dụng của t-SNE")
    st.write("""
    t-SNE được sử dụng rộng rãi trong nhiều lĩnh vực, bao gồm:  
    - **Trực quan hóa dữ liệu**: Biến các tập dữ liệu phức tạp (hình ảnh, văn bản, dữ liệu sinh học) thành biểu đồ 2D dễ hiểu.  
    - **Khám phá cụm**: Phát hiện các nhóm dữ liệu mà không cần biết trước số lượng cụm.  
    - **Phân tích đặc trưng trong học máy**: Hiểu cấu trúc của không gian đặc trưng để cải thiện mô hình.
    """)

    # Phần 6: Lưu ý khi sử dụng t-SNE
    st.header("6. Lưu ý khi sử dụng t-SNE")
    st.write("""
    Khi áp dụng t-SNE, bạn cần lưu ý:  
    - **Tham số perplexity**: Quyết định số lượng hàng xóm được xem xét, thường nằm trong khoảng 5-50 tùy kích thước dữ liệu.  
    - **Không bảo toàn khoảng cách toàn cục**: t-SNE ưu tiên cấu trúc cục bộ, nên khoảng cách giữa các cụm có thể không phản ánh đúng thực tế.  
    - **Chi phí tính toán**: Với dữ liệu lớn, t-SNE có thể chậm; hãy cân nhắc sử dụng phiên bản tối ưu như **Barnes-Hut t-SNE**.  
    - **Tính ngẫu nhiên**: Kết quả có thể thay đổi giữa các lần chạy do khởi tạo ngẫu nhiên, trừ khi cố định seed.
    """)
