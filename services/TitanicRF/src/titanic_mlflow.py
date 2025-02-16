import mlflow
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

from mlflow.models import infer_signature


# Bật MLflow để theo dõi thí nghiệm
exp_id = mlflow.set_experiment("Titanic_Data_Processing")


def main_mlflow():
    with mlflow.start_run(experiment_id=exp_id.experiment_id):
        # 1. Load dataset từ file tải lên
        file_path = "./services/TitanicRF/data/titanic.csv"
        df = pd.read_csv(file_path)
        mlflow.log_param("Dataset_size", df.shape[0])

        # 2. Kiểm tra missing values
        missing_values = df.isnull().sum().to_dict()
        print(f"Missing values trước xử lý: {missing_values}")
        mlflow.log_dict(missing_values, "missing_values.json")

        # 3. Điền giá trị thiếu
        df['Age'].fillna(df['Age'].median(), inplace=True)
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
        df.drop(columns=['Cabin'], inplace=True)
        mlflow.log_param("missing_values_filled", True)

        # 4. Encode các biến phân loại
        label_encoder = LabelEncoder()
        df['Sex'] = label_encoder.fit_transform(df['Sex'])
        df['Embarked'] = label_encoder.fit_transform(df['Embarked'])
        mlflow.log_param("encoded_columns", ["Sex", "Embarked"])

        # 5. Loại bỏ các cột không cần thiết
        df.drop(columns=['Name', 'Ticket'], inplace=True)
        print("[INFO] Đã loại bỏ các cột không cần thiết: Name, Ticket")

        # 6. Chuẩn hóa dữ liệu số
        scaler = StandardScaler()
        numerical_features = ["Age", "Fare"]
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        mlflow.log_param("scaled_features", numerical_features)

        # 6.1. Lưu dataset đã xử lý thành file CSV
        processed_file = "./services/TitanicRF/data/new_titanic.csv"
        df.to_csv(processed_file, index=False)
        print(f"[INFO] Dữ liệu đã xử lý được lưu tại {processed_file}")

        # 6.2. Ghi log file đã xử lý vào MLflow
        mlflow.log_artifact(processed_file)
        print("[INFO] Quá trình tiền xử lý hoàn thành và được ghi log vào MLflow.")

        # 7. Chia tập dữ liệu thành train (70%), valid (15%), test (15%)
        X = df.drop(columns=['Survived'])
        y = df['Survived']

        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, random_state=42, shuffle=True)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, shuffle=True)

        # Ghi lại thông tin chia cắt vào MLflow
        mlflow.log_param("random_state", 42)
        mlflow.log_param('shuffle', True)

        mlflow.log_param("training_size", X_train.shape[0])
        mlflow.log_param("valid_size", X_val.shape[0])
        mlflow.log_param("test_size", X_test.shape[0])

        # mlflow.log_metric("Train_sample", len(X_train))
        # mlflow.log_metric("Test_sample", len(X_test))
        # mlflow.log_metric("Valid_sample", len(X_val))

        # 8. Lưu dataset đã chia
        train_df = pd.concat([X_train, y_train], axis=1)
        valid_df = pd.concat([X_val, y_val], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_file = "./services/TitanicRF/data/train_titanic.csv"
        valid_file = "./services/TitanicRF/data/valid_titanic.csv"
        test_file = "./services/TitanicRF/data/test_titanic.csv"

        train_df.to_csv(train_file, index=False)
        valid_df.to_csv(valid_file, index=False)
        test_df.to_csv(test_file, index=False)

        mlflow.log_artifact(train_file)
        mlflow.log_artifact(valid_file)
        mlflow.log_artifact(test_file)
        print(" Đã chia thành công file train-valid-test")

        # 9. Huấn luyện mô hình Random Forest với Cross Validation
        # mlflow.sklearn.autolog()

        rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)

        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_leaf", 5)
        mlflow.log_param("k_fold", 5)

        scores = cross_val_score(
            rf_model, X_train, y_train, cv=5, scoring='accuracy')
        for fold_idx, score in enumerate(scores, 1):
            print(
                f"[INFO] Cross-validation fold {fold_idx}: Accuracy = {score:.4f}")
        mean_score = scores.mean()
        print(
            f"[INFO] Accuracy trung bình trên tập train (cross-validation): {mean_score:.4f}")
        mlflow.log_metric("cross_valid_avg_accuracy", mean_score)

        # Kết quả Huấn luyện mô hình trên tập train và kiểm thử trên tập validation
        rf_model.fit(X_train, y_train)
        y_train_pred = rf_model.predict(X_train)
        signature = infer_signature(X_train, y_train)

        train_accuracy = accuracy_score(y_train, y_train_pred)
        print(f"[INFO] Accuracy trên tập train: {train_accuracy:.4f}")
        mlflow.log_metric("train_accuracy", train_accuracy)

        y_val_pred = rf_model.predict(X_val)
        val_accuracy = accuracy_score(y_val, y_val_pred)
        print(f"[INFO] Accuracy trên tập validation: {val_accuracy:.4f}")
        mlflow.log_metric("val_accuracy", val_accuracy)

        y_test_pred = rf_model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"[INFO] Accuracy trên tập test: {test_accuracy:.4f}")
        mlflow.log_metric("test_accuracy", test_accuracy)

        # 10. Lưu mô hình đã huấn luyện
        # mlflow.sklearn.log_model(rf_model, "RandomForestModel")
        mlflow.sklearn.log_model(
            sk_model=rf_model,
            artifact_path="sklearn-model",
            signature=signature,
            registered_model_name="RandomForestModel",
        )

        print(" Đã hoàn thành huấn luyện mô hình Random Forest và ghi log vào MLflow")


if __name__ == "__main__":
    main_mlflow()
