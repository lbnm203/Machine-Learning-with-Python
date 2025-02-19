import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error, r2_score, accuracy_score, mean_squared_error


from services.Linear_Regression.utils.preprocess import preprocess_data
from services.Linear_Regression.utils.org_data import visualize_org_data
from services.Linear_Regression.utils.training_ln import training
from services.Linear_Regression.utils.theory_ln import theory_linear
from services.Linear_Regression.utils.demo_st import demo_app

# Khởi tạo MLflow
# mlflow.set_tracking_uri("http://127.0.0.1:5000")
# # mlflow.create_experiment("Titanic_Survival_Prediction")
# mlflow.set_experiment("Titanic_Survival_Prediction")


# @st.cache_resource
def main():
    st.title("Titanic Survival Prediction with Linear Regression")

    url = "./services/TitanicRF/data/titanic.csv"
    data = pd.read_csv(url)

    data_org, data_preprocess, theory, train_process, demo = st.tabs(
        ["Dữ liệu gốc", "Dữ liệu được tiền xử lý", "Thông tin", "Huấn luyện", "Demo"])

    with data_org:
        visualize_org_data(data)

    with data_preprocess:
        st.markdown("### ⚙️ Quá trình tiền xử lý dữ liệu")
        st.markdown("""
            - Xử lý các giá trị thiếu
                - **Age**: Điền giá trị trung bình cho các giá trị bị thiếu.
                - **Fare**: Điền giá trị trung bình cho các giá trị bị thiếu.
                - **Embarked**: Điền giá trị phổ biến nhất (mode) cho các giá trị bị thiếu.

            - Mã hóa kiểu dữ liệu dạng chữ về dạng số để mô hình có thể xử lý:
                - **Sex**: Chuyển đổi giá trị của cột Sex từ dạng categorical sang dạng numerical ```(0: male, 1: female)```.
                - **Embarked**: tương tự ở phần này ta sẽ tiến hành chuyển các giá trị về dạng numerical ```(1: S, 2: C, 3: Q)```

            - Tiến hành drop 3 đặc trưng **Cabin, Ticket, Name** vì sau khi thống kê cho thấy đặc trưng **Ticket** và **Name** không ảnh hưởng hay không giúp ích đến kết quả dự đoán,
            bỏ qua đặc trưng **Cabin** vì giá trị thiếu chiếm đến ~80% (687/891)

            - Chuẩn hóa các dữ liệu ['Age'], ['Fare'], ['Pclass'], ['SibSp'], ['Parch'] cùng một khoảng giá trị tương ứng trong tập dữ liệu để phù hợp cho việc tính toán của mô hình
        """)

        X, y, preprocessor, df = preprocess_data(data)
        st.markdown("### 📝 Tập dữ liệu sau tiền xử lý")
        st.dataframe(df)

    with theory:
        theory_linear()

    # Chia dữ liệu
    with train_process:
        training(df)

    # Phần demo dự đoán
    with demo:
        model_type = st.selectbox("Select Model Type", [
                                  "Multiple Regression", "Polynomial Regression"])

        if model_type == "Polynomial Regression":
            degree = 2  # Define the degree variable
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('poly', PolynomialFeatures(degree=degree, interaction_only=False)),
                ('regressor', LinearRegression())
            ])
        else:
            model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', LinearRegression())
            ])

        demo_app(X, y, model, model_type)


if __name__ == "__main__":
    main()
