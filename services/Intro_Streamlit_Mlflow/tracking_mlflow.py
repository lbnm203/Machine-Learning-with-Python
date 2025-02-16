import mlflow
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import ElasticNet
from mlflow.models import infer_signature


def eval_metrics(actual, pred):
    mae = mean_absolute_error(actual, pred)
    mse = mean_squared_error(actual, pred)
    return mae, mse


if __name__ == "__main__":
    # Load data
    data = pd.read_csv(
        "./services/Intro_Streamlit_Mlflow/data/winequality-red.csv")

    # Split data
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    # Split features and target
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train["quality"]
    test_y = test["quality"]

    exp = mlflow.set_experiment("Wine_Quality_Prediction")
    
    with mlflow.start_run(experiment_id=exp.experiment_id):
        # # Log dataset size
        mlflow.log_param("train_size", train_x.shape[0])
        mlflow.log_param("test_size", test_x.shape[0])

        # Train model
        alpha = 0.5
        l1_ratio = 0.5
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        # Predict
        predicted_qualities = lr.predict(test_x)
        signature = infer_signature(test_x, predicted_qualities)

        # Evaluate
        mae, mse = eval_metrics(test_y, predicted_qualities)

        # Log metrics
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse", mse)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="ElasticNet_Wine_Model",
            signature=signature,
            registered_model_name="ElasticNet_Wine_Model",
        )

        print(" Đã hoàn thành huấn luyện mô hình Random Forest và ghi log vào MLflow")
