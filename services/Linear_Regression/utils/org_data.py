import streamlit as st
import pandas as pd
from scipy.stats import zscore


def visualize_org_data(data):
    # st.markdown("### 📄 Tập dữ liệu gốc")
    # st.dataframe(data)
    # st.markdown("""
    #     ---
    #     ### 🗒️ Thông tin của tập dữ liệu:
    #     - ```PassengerId```: ID của hành khách (không cần thiết cho mô hình).

    #     - ```Survived```: Biến mục tiêu (0 = Không sống sót, 1 = Sống sót).

    #     - ```Pclass```: Hạng vé (1, 2, 3).

    #     - ```Name```: Tên hành khách.

    #     - ```Sex```: Giới tính (male, female).

    #     - ```Age```: Tuổi

    #     - ```SibSp```: Số anh chị em hoặc vợ/chồng đi cùng.

    #     - ```Parch```: Số cha mẹ hoặc con cái đi cùng.

    #     - ```Ticket```: Số vé.

    #     - ```Fare```: Giá vé

    #     - ```Cabin```: Số phòng

    #     - ```Embarked```: Cảng lên tàu (C = Cherbourg, Q = Queenstown, S = Southampton).
    # """)

    # st.write("Kích thức tập dữ liệu:", data.shape)
    # st.markdown("---")
    # st.markdown("### 📊 Thống kê tập dữ liệu")
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.write("##### Dữ liệu bị thiếu:")
    #     missing_values = data.isnull().sum()
    #     st.write(missing_values)

    # with col2:
    #     st.write("##### Số dữ liệu trùng lặp:")
    #     duplicated_data = data.duplicated().sum()
    #     st.write(duplicated_data)

    # st.write("##### Phân phối tập dữ liệu:")
    # st.write(data.describe().T)
    st.markdown("### 📄 Tập dữ liệu gốc")
    st.dataframe(data)
    st.markdown("""
        ---
        ### 🗒️ Thông tin của tập dữ liệu:
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
            ```PassengerId```: ID của hành khách 

            ```Survived```: Biến mục tiêu (0 = Không sống sót, 1 = Sống sót).

            ```Pclass```: Hạng vé (1, 2, 3).

            ```Name```: Tên hành khách.

            ```Sex```: Giới tính (male, female).

            ```Age```: Tuổi
        """)

    with col2:
        st.markdown("""
            ```SibSp```: Số anh chị em hoặc vợ/chồng đi cùng.

            ```Parch```: Số cha mẹ hoặc con cái đi cùng.

            ```Ticket```: Số vé.

            ```Fare```: Giá vé

            ```Cabin```: Số phòng

            ```Embarked```: Cảng lên tàu (C = Cherbourg, Q = Queenstown, S = Southampton).

        """)

    st.markdown("---")
    st.markdown("### 📊 Thống kê tập dữ liệu")
    col1, col2 = st.columns(2)
    with col1:
        st.write("##### Dữ liệu bị thiếu:")
        missing_values = data.isnull().sum()
        outlier_count = {
            col: (abs(zscore(data[col], nan_policy='omit')) > 3).sum()
            for col in data.select_dtypes(include=['number']).columns
        }
        miss_table = pd.DataFrame({
            'Giá trị bị thiếu': missing_values,
            'Outlier': [outlier_count.get(col, 0) for col in data.columns]
        })
        st.table(miss_table)

    with col2:
        st.write("##### Số dữ liệu trùng lặp:")
        duplicated_data = data.duplicated().sum()
        st.write(duplicated_data)

    st.write("##### Phân phối tập dữ liệu:")
    st.table(data.describe().round(2))
