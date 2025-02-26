import streamlit as st


def explain_kmeans():
    st.header("Thuật toán KMeans")
    st.write("""
    **KMeans** là một thuật toán phân cụm không giám sát, dùng để chia dữ liệu thành **K** cụm. Các điểm trong cùng một cụm sẽ gần nhau hơn so với các điểm ở các cụm khác.

    - **Mục đích**: Tìm các nhóm tự nhiên trong dữ liệu mà không cần nhãn.
    - **Ứng dụng**: Phân đoạn khách hàng, nén ảnh, nhận diện mẫu, v.v.
    """)

    st.subheader("Các khái niệm cơ bản")
    st.write("""
    - **Centroid**: Điểm đại diện cho tâm của cụm, tính bằng trung bình của các điểm trong cụm.
    - **Khoảng cách Euclidean**: Đo khoảng cách giữa hai điểm, giúp xác định điểm nào gần centroid nào nhất.
    - **Hàm chi phí**: Tổng bình phương khoảng cách từ các điểm đến centroid, được KMeans cố gắng giảm xuống nhỏ nhất.
    """)

    st.subheader("Các bước của thuật toán KMeans")
    st.write("""
    1. **Khởi tạo centroid**:
       - Chọn ngẫu nhiên **K** điểm làm centroid ban đầu.
       - Có thể dùng KMeans++ để cải thiện lựa chọn ban đầu.

    2. **Gán nhãn cho các điểm**:
       - Tính khoảng cách từ mỗi điểm đến các centroid.
       - Gán điểm vào cụm có centroid gần nhất.

    3. **Cập nhật centroid**:
       - Tính lại centroid bằng trung bình của các điểm trong cụm.

    4. **Lặp lại**:
       - Tiếp tục bước 2 và 3 cho đến khi centroid ổn định hoặc đạt số lần lặp tối đa.
    """)

    st.subheader("Ví dụ minh họa")
    st.write("""
    Giả sử có một tập dữ liệu 2D với các điểm phân bố thành hai nhóm. KMeans sẽ phân cụm như sau:

    - **Bước 1**: Chọn ngẫu nhiên 2 centroid (K=2).
    - **Bước 2**: Gán mỗi điểm vào cụm có centroid gần nhất.
    - **Bước 3**: Cập nhật centroid dựa trên trung bình của các điểm trong cụm.
    - **Bước 4**: Lặp lại đến khi centroid không thay đổi.
    """)

    st.subheader("Ưu và nhược điểm của KMeans")
    st.write("""
    **Ưu điểm**:
    - Dễ hiểu và triển khai.
    - Hiệu quả với tập dữ liệu lớn.
    - Tốc độ nhanh.

    **Nhược điểm**:
    - Phải chọn trước số cụm **K**.
    - Nhạy cảm với centroid ban đầu.
    - Chỉ phù hợp với cụm hình cầu, không tốt cho dữ liệu phức tạp hoặc nhiều nhiễu.
    """)


def explain_dbscan():
    # Tiêu đề chính
    st.header("Thuật toán DBSCAN")
    st.write("""
    **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** là một thuật toán phân cụm dựa trên mật độ. Nó nhóm các điểm dữ liệu gần nhau thành các cụm và xác định các điểm nhiễu (noise). Điểm đặc biệt của DBSCAN là không cần xác định trước số lượng cụm và có thể phát hiện các cụm có hình dạng bất kỳ.
    """)

    # Phần 1: Các khái niệm cơ bản
    st.subheader("Các khái niệm cơ bản")
    st.write("""
    - **Epsilon (ε)**: Bán kính của vùng lân cận xung quanh một điểm.
    - **Min_samples**: Số lượng điểm tối thiểu trong vùng lân cận ε để một điểm được coi là điểm lõi (core point).
    - **Điểm lõi (core point)**: Điểm có ít nhất `min_samples` điểm trong vùng lân cận ε.
    - **Điểm biên (border point)**: Điểm không phải là lõi nhưng nằm trong vùng lân cận của một điểm lõi.
    - **Điểm nhiễu (noise point)**: Điểm không phải lõi và không nằm trong vùng lân cận của bất kỳ điểm lõi nào.
    """)

    # Phần 2: Cách thức hoạt động
    st.subheader("Cách thức hoạt động của DBSCAN")
    st.write("""
    1. **Chọn tham số**: Xác định giá trị cho ε và min_samples.
    2. **Phân loại các điểm**:
        - Bắt đầu với một điểm chưa được thăm.
        - Nếu điểm là lõi, tạo một cụm mới và thêm tất cả các điểm trong vùng lân cận của nó vào cụm.
        - Mở rộng cụm bằng cách kiểm tra đệ quy các điểm lân cận của các điểm lõi mới.
    3. **Xử lý điểm biên và nhiễu**:
        - Điểm không phải lõi nhưng nằm trong vùng lân cận của lõi được gán vào cụm (biên).
        - Điểm không thuộc bất kỳ cụm nào được coi là nhiễu.
    4. **Lặp lại**: Tiếp tục cho đến khi tất cả các điểm được thăm.
    """)

    # Phần 3: Ví dụ minh họa
    st.subheader("Ví dụ minh họa")
    col1, col2 = st.columns(2)
    with col1:
        st.write("""
        Giả sử bạn có một tập dữ liệu 2D với các điểm phân bố thành hai nhóm dày đặc và một số điểm rải rác:
        - Với ε = 1.0 và min_samples = 4, DBSCAN sẽ:
            - Xác định các điểm lõi trong vùng dày đặc.
            - Mở rộng cụm từ các điểm lõi.
            - Gán các điểm gần cụm thành điểm biên.
            - Đánh dấu các điểm rải rác là nhiễu.
        """)
    with col2:
        st.image("./services/mnist_clustering/assest/dbscan_1.jpeg",
                 caption="Ví dụ minh họa của DBSCAN")

    # Phần 4: Ưu và nhược điểm
    st.subheader("Ưu và nhược điểm của DBSCAN")
    st.write("""
    **Ưu điểm**:
    - Không cần xác định số lượng cụm trước.
    - Có thể phát hiện cụm có hình dạng bất kỳ.
    - Hiệu quả trong việc loại bỏ nhiễu.

    **Nhược điểm**:
    - Khó chọn tham số ε và min_samples phù hợp.
    - Hiệu suất kém với dữ liệu cao chiều nếu không giảm chiều.
    - Không phù hợp nếu các cụm có mật độ khác nhau.
    """)
