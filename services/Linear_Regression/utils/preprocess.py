import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def preprocess_data(df):
    """Tiền xử lý dữ liệu Titanic."""

    df = df.copy()

    # 1. Xử lý missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Drop 3 đặc trưng Ticket, Cabin, Name
    df.drop(columns=['Ticket', 'Cabin', 'Name'], inplace=True)

    # 4. Encode cho các biến phân loại
    df['Embarked'] = df['Embarked'].map(
        {'S': 1, 'C': 2, 'Q': 3}).astype('Int64')
    df["Sex"] = df["Sex"].map({'male': 0, 'female': 1}).astype('Int64')

    # 3. Chuẩn hóa các cột số
    scaler = StandardScaler()
    numerical_features = ['Age', 'Fare', 'Pclass', 'SibSp', 'Parch', 'Sex', 'Embarked', 'PassengerId', 'Survived']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])


    # features = [col for col in df.columns if col != 'Survived']
    # target = 'Survived'
    # X = df[features]
    # y = df[target]
    X = df.drop(columns=['Survived'])
    y = df['Survived']

    # Định nghĩa preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [
                'Age', 'Fare', 'Pclass', 'SibSp', 'Parch']),
            ('cat', OneHotEncoder(), ['Embarked', 'Sex'])
        ]
    )

    return X, y, preprocessor, df
