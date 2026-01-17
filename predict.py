"""Simple training + prediction script.

Usage:
  python predict.py

This will train the linear regression model and print evaluation metrics.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def main():
    df = pd.read_csv("data/HousePricePrediction.csv")

    # Preprocessing
    if "Id" in df.columns:
        df = df.drop(["Id"], axis=1)

    df["SalePrice"] = df["SalePrice"].fillna(df["SalePrice"].mean())
    df = df.dropna()

    # One-hot encode selected columns
    cols = ["MSZoning", "LotConfig", "BldgType", "Exterior1st"]
    df = pd.get_dummies(df, columns=cols, drop_first=True)

    X = df.drop(["SalePrice"], axis=1)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"R2 Score: {r2:.3f}")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")


if __name__ == "__main__":
    main()
