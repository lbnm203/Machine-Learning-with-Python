import streamlit as st
import pandas as pd
import mlflow
import mlflow.tracking
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score


# Load dữ liệu từ file CSV
def load_dataset(file_path):
    return pd.read_csv(file_path)


# def get_mlflow_runs(experiment_name):
#     client = mlflow.tracking.MlflowClient()
#     experiment = client.get_experiment_by_name(experiment_name)
#     if experiment:
#         experiment_id = experiment.experiment_id
#         return client.search_runs(experiment_id, order_by=["metrics.val_accuracy DESC"])
#     return None


def display_best_run(best_run):
    st.markdown("---")
    st.write("### Mô tả các bước thực hiện:")
    st.markdown("""
    - **1. Load dataset từ file tải lên**
    - **2. Tiền xử lý dữ liệu**
        - Kiểm tra missing values
        - Điền giá trị thiếu
        - Encode các biến phân loại
        - Loại bỏ các cột không cần thiết
        - Chuẩn hóa dữ liệu số
    - **3. Chia tập dữ liệu thành** 
        - Train (70%)
        - Validation (15%)
        - Test (15%)
    - **4. Huấn luyện mô hình Random Forest với Cross Validation** 
        - Sử dụng Cross Validation (k-fold) để đánh giá hiệu suất mô hình trên nhiều tập con.
    - **5. Đánh giá mô hình**
                
    """)

    process = Image.open("./services/TitanicRF/result/flow_process.png")
    st.image(process, caption="Minh họa quá trình thực hiện",
             use_column_width=True)

    st.markdown("---")
    st.markdown("""
    #### **1. Load dataset từ file tải lên** 
    """)
    df = load_dataset("./services/TitanicRF/data/titanic.csv")
    st.dataframe(df.head(10))

    st.write("##### Thống kê dữ liệu:")
    st.write(df.describe())

    st.markdown("---")
    st.markdown("""
    #### **2. Kiểm tra missing values và dữ liệu bị trùng lặp** 
    """)
    col1, col2 = st.columns(2)
    with col1:
        st.write("##### Missing values:")
        missing_values = df.isnull().sum()
        st.write(missing_values)

    with col2:
        st.write("##### Duplicate values:")
        duplicated_data = df.duplicated().sum()
        st.write(duplicated_data)

    # st.write(df.isnull().sum())

    st.markdown("---")
    st.markdown("""
    #### **3. Điền giá trị thiếu** 
    """)
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(columns=['Cabin'], inplace=True)
    st.dataframe(df.head())

    st.markdown("---")
    st.markdown("""
    #### **4. Encode các biến phân loại [Sex], [Embarked]** 
    """)
    df_encoded = df.copy()
    df_encoded['Sex'] = df_encoded['Sex'].astype('category').cat.codes
    df_encoded['Embarked'] = df_encoded['Embarked'].astype(
        'category').cat.codes
    st.dataframe(df_encoded.head())

    st.markdown("---")
    st.markdown("""
    #### **5. Loại bỏ các cột không cần thiết [Name], [Ticket]** 
    """)
    df_encoded.drop(columns=['Name', 'Ticket'], inplace=True)
    st.dataframe(df_encoded.head())

    st.markdown("---")
    st.markdown("""
    #### **6. Chuẩn hóa dữ liệu số [Age], [Fare]** 
    """)

    scaler = StandardScaler()
    numerical_features = ['Age', 'Fare']
    df_encoded[numerical_features] = scaler.fit_transform(
        df_encoded[numerical_features])
    st.dataframe(df_encoded.head())

    st.markdown("---")
    st.markdown("""
    #### **7. Chia tập dữ liệu thành train (70%), valid (15%), test (15%)**
    """)

    X = df_encoded.drop(columns=['Survived'])
    y = df_encoded['Survived']

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42)

    # Hiển thị biểu đồ kích thước các tập dữ liệu
    fig, ax = plt.subplots()
    ax.bar(['Train', 'Validation', 'Test'], [X_train.shape[0],
           X_val.shape[0], X_test.shape[0]], color=['blue', 'green', 'red'])
    ax.set_ylabel("Số lượng mẫu")
    ax.set_title("Kích thước các tập dữ liệu")
    for i, v in enumerate([X_train.shape[0], X_val.shape[0], X_test.shape[0]]):
        ax.text(i, v + 5, str(v), ha='center', fontsize=10, fontweight='bold')
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("""
    #### **8. Huấn luyện mô hình Random Forest với Cross Validation**
    """)

    # st.write("Kiểm tra nội dung best_run:")
    # st.write(best_run)

    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
    scores = cross_val_score(rf_model, X_train, y_train,
                             cv=5, scoring='accuracy')

    st.write("##### Tham số mô hình:")
    st.markdown(f"- **n_estimators:** {best_run.data.params['n_estimators']}")
    st.markdown(f"- **max_depth:** {best_run.data.params['max_depth']}")
    st.markdown(
        f"- **min_samples_leaf:** {best_run.data.params['min_samples_leaf']}")
    st.markdown(f"- **k-fold:** {best_run.data.params['k_fold']}")

    st.markdown("---")
    st.write("##### Kết quả đánh giá mô hình:")
    st.write(
        f"- **Train Accuracy:** {best_run.data.metrics['train_accuracy']:.4f}")
    st.write(
        f"- **Validation Accuracy:** {best_run.data.metrics['val_accuracy']:.4f}")
    st.write(
        f"- **Test Accuracy:** {best_run.data.metrics['test_accuracy']:.4f}")
    st.write(
        f"- **Cross-validation Accuracy:** {best_run.data.metrics['cross_valid_avg_accuracy']:.4f}")
    st.markdown("---")

    # Hiển thị biểu đồ accuracy Cross-validation qua từng fold
    fig, ax = plt.subplots()
    ax.plot(range(1, 6), scores, marker='o', linestyle='-', color='c')
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Cross-validation qua từng fold")
    for i, v in enumerate(scores):
        ax.text(i + 1, v + 0.005, f"{v:.4f}",
                ha='center', fontsize=10, fontweight='bold')
    st.pyplot(fig)


def plot_accuracy_graph(runs):
    train_acc = [run.data.metrics['train_accuracy'] for run in runs]
    val_acc = [run.data.metrics['val_accuracy'] for run in runs]
    test_acc = [run.data.metrics['test_accuracy'] for run in runs]

    fig, ax = plt.subplots()
    categories = ['Train Accuracy', 'Validation Accuracy', 'Test Accuracy']
    values = [train_acc[-1], val_acc[-1], test_acc[-1]]
    ax.bar(categories, values, color=['blue', 'green', 'red'])
    ax.set_ylabel("Accuracy")
    ax.set_title("So sánh Accuracy giữa Train, Validation và Test")
    for i, v in enumerate(values):
        ax.text(i, v + 0.005, f"{v:.4f}", ha='center',
                fontsize=10, fontweight='bold')
    st.pyplot(fig)


def display_run_list(runs):
    st.markdown("---")
    st.subheader(" Danh sách các lần chạy trên MLflow")
    df_runs = pd.DataFrame({
        "Run ID": [run.info.run_id for run in runs],
        "Train Accuracy": [run.data.metrics['train_accuracy'] for run in runs],
        "Validation Accuracy": [run.data.metrics['val_accuracy'] for run in runs],
        "Test Accuracy": [run.data.metrics['test_accuracy'] for run in runs]
    })
    st.dataframe(df_runs)


def main():
    st.title(" ✨ Xử Lý Dữ Liệu Titanic Kết Hợp Huấn Luyện Trên Mô Hình Random Forest")

    mlflow.set_tracking_uri("mlruns")
    experiment_name = "Titanic_Data_Processing"
    mlflow.set_experiment(experiment_name)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
        runs = client.search_runs(experiment_id, order_by=[
                                  "metrics.val_accuracy DESC"])
        if not runs:
            st.error("Không tìm thấy thí nghiệm MLflow")
        else:
            st.success("Kết nối thành công MLflow")
            display_best_run(runs[0])
            plot_accuracy_graph(runs)
            display_run_list(runs)
    return None

    # # mlflow.set_tracking_uri("mlruns")
    # # experiment_name = "Titanic_Data_Processing"
    # runs = get_mlflow_runs(experiment_name)

    # if not runs:
    #     st.error("Không tìm thấy thí nghiệm MLflow")
    #     return

    # display_best_run(runs[0])
    # plot_accuracy_graph(runs)
    # display_run_list(runs)


if __name__ == "__main__":
    main()

# # Load dữ liệu từ file CSV
# def load_dataset(file_path):
#     return pd.read_csv(file_path)


# def display_best_run():
#     st.markdown("---")
#     st.write("### Mô tả các bước thực hiện:")
#     st.markdown("""
#     - **1. Load dataset từ file tải lên**
#     - **2. Tiền xử lý dữ liệu**
#         - Kiểm tra missing values
#         - Điền giá trị thiếu
#         - Encode các biến phân loại
#         - Loại bỏ các cột không cần thiết
#         - Chuẩn hóa dữ liệu số
#     - **3. Chia tập dữ liệu thành**
#         - Train (70%)
#         - Validation (15%)
#         - Test (15%)
#     - **4. Huấn luyện mô hình Random Forest với Cross Validation**
#         - Sử dụng Cross Validation (k-fold) để đánh giá hiệu suất mô hình trên nhiều tập con.
#     - **5. Đánh giá mô hình**

#     """)

#     process = Image.open("./services/TitanicRF/result/flow_process.png")
#     st.image(process, caption="Minh họa quá trình thực hiện",
#              use_container_width=True)

#     st.markdown("---")
#     st.markdown("""
#     #### **1. Load dataset từ file tải lên**
#     """)
#     df = load_dataset("./services/TitanicRF/data/titanic.csv")
#     st.dataframe(df.head(10))

#     st.write("##### Thống kê dữ liệu:")
#     st.write(df.describe())

#     st.markdown("---")
#     st.markdown("""
#     #### **2. Kiểm tra missing values và dữ liệu bị trùng lặp**
#     """)
#     col1, col2 = st.columns(2)
#     with col1:
#         st.write("##### Missing values:")
#         missing_values = df.isnull().sum()
#         st.write(missing_values)

#     with col2:
#         st.write("##### Duplicate values:")
#         duplicated_data = df.duplicated().sum()
#         st.write(duplicated_data)

#     # st.write(df.isnull().sum())

#     st.markdown("---")
#     st.markdown("""
#     #### **3. Điền giá trị thiếu**
#     """)
#     df['Age'].fillna(df['Age'].median(), inplace=True)
#     df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
#     df.drop(columns=['Cabin'], inplace=True)
#     st.dataframe(df.head())

#     st.markdown("---")
#     st.markdown("""
#     #### **4. Encode các biến phân loại [Sex], [Embarked]**
#     """)
#     df_encoded = df.copy()
#     df_encoded['Sex'] = df_encoded['Sex'].astype('category').cat.codes
#     df_encoded['Embarked'] = df_encoded['Embarked'].astype(
#         'category').cat.codes
#     st.dataframe(df_encoded.head())

#     st.markdown("---")
#     st.markdown("""
#     #### **5. Loại bỏ các cột không cần thiết [Name], [Ticket]**
#     """)
#     df_encoded.drop(columns=['Name', 'Ticket'], inplace=True)
#     st.dataframe(df_encoded.head())

#     st.markdown("---")
#     st.markdown("""
#     #### **6. Chuẩn hóa dữ liệu số [Age], [Fare]**
#     """)

#     scaler = StandardScaler()
#     numerical_features = ['Age', 'Fare']
#     df_encoded[numerical_features] = scaler.fit_transform(
#         df_encoded[numerical_features])
#     st.dataframe(df_encoded.head())

#     st.markdown("---")
#     st.markdown("""
#     #### **7. Chia tập dữ liệu thành train (70%), valid (15%), test (15%)**
#     """)

#     X = df_encoded.drop(columns=['Survived'])
#     y = df_encoded['Survived']

#     X_train, X_temp, y_train, y_temp = train_test_split(
#         X, y, test_size=0.3, random_state=42)
#     X_val, X_test, y_val, y_test = train_test_split(
#         X_temp, y_temp, test_size=0.5, random_state=42)

#     # Hiển thị biểu đồ kích thước các tập dữ liệu
#     fig, ax = plt.subplots()
#     ax.bar(['Train', 'Validation', 'Test'], [X_train.shape[0],
#            X_val.shape[0], X_test.shape[0]], color=['blue', 'green', 'red'])
#     ax.set_ylabel("Số lượng mẫu")
#     ax.set_title("Kích thước các tập dữ liệu")
#     for i, v in enumerate([X_train.shape[0], X_val.shape[0], X_test.shape[0]]):
#         ax.text(i, v + 5, str(v), ha='center', fontsize=10, fontweight='bold')
#     st.pyplot(fig)

#     st.markdown("---")
#     st.markdown("""
#     #### **8. Huấn luyện mô hình Random Forest với Cross Validation**
#     """)

#     # st.write("Kiểm tra nội dung best_run:")
#     # st.write(best_run)

#     rf_model = RandomForestClassifier(
#         n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)
#     scores = cross_val_score(rf_model, X_train, y_train,
#                              cv=5, scoring='accuracy')

#     # Huấn luyện mô hình trên tập train
#     rf_model.fit(X_train, y_train)

#     # Tính toán accuracy trên các tập dữ liệu
#     train_accuracy = rf_model.score(X_train, y_train)
#     val_accuracy = rf_model.score(X_val, y_val)
#     test_accuracy = rf_model.score(X_test, y_test)
#     cross_valid_avg_accuracy = scores.mean()

#     st.write("##### Tham số mô hình:")
#     st.markdown(f"- **n_estimators:** 100")
#     st.markdown(f"- **max_depth:** 10")
#     st.markdown(f"- **min_samples_leaf:** 10")
#     st.markdown(f"- **k-fold:** 5")
#     st.markdown(f"- **random_state:** 42")

#     st.markdown("---")
#     st.write("##### Kết quả đánh giá mô hình:")
#     rf_model.fit(X_train, y_train)
#     train_accuracy = rf_model.score(X_train, y_train)
#     val_accuracy = rf_model.score(X_val, y_val)
#     test_accuracy = rf_model.score(X_test, y_test)
#     cross_valid_avg_accuracy = scores.mean()

#     st.write(f"- **Train Accuracy:** {train_accuracy:.4f}")
#     st.write(f"- **Validation Accuracy:** {val_accuracy:.4f}")
#     st.write(f"- **Test Accuracy:** {test_accuracy:.4f}")
#     st.write(
#         f"- **Cross-validation Accuracy:** {cross_valid_avg_accuracy:.4f}")
#     st.markdown("---")

#     # Hiển thị biểu đồ accuracy Cross-validation qua từng fold
#     fig, ax = plt.subplots()
#     ax.plot(range(1, 6), scores, marker='o', linestyle='-', color='c')
#     ax.set_xlabel("Fold")
#     ax.set_ylabel("Accuracy")
#     ax.set_title("Accuracy Cross-validation qua từng fold")
#     for i, v in enumerate(scores):
#         ax.text(i + 1, v + 0.005, f"{v:.4f}",
#                 ha='center', fontsize=10, fontweight='bold')
#     st.pyplot(fig)

#     # Trả về các giá trị accuracy để sử dụng ở nơi khác
#     return train_accuracy, val_accuracy, test_accuracy


# def plot_accuracy_graph(train_accuracy, val_accuracy, test_accuracy):
#     fig, ax = plt.subplots()
#     categories = ['Train Accuracy', 'Validation Accuracy', 'Test Accuracy']
#     values = [train_accuracy, val_accuracy, test_accuracy]
#     ax.bar(categories, values, color=['blue', 'green', 'red'])
#     ax.set_ylabel("Accuracy")
#     ax.set_title("So sánh Accuracy giữa Train, Validation và Test")
#     for i, v in enumerate(values):
#         ax.text(i, v + 0.005, f"{v:.4f}", ha='center',
#                 fontsize=10, fontweight='bold')
#     st.pyplot(fig)


# def main():
#     st.title(" ✨ Xử Lý Dữ Liệu Titanic Kết Hợp Huấn Luyện Trên Mô Hình Random Forest")

#     train_accuracy, val_accuracy, test_accuracy = display_best_run()
#     plot_accuracy_graph(train_accuracy, val_accuracy, test_accuracy)


# if __name__ == "__main__":
#     main()
