import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import os
import argparse

mlflow.sklearn.autolog()

def train_model(preprocessed_data_path):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Loading preprocessed data from: {preprocessed_data_path}")

    df = pd.read_csv(preprocessed_data_path)

    X = df.drop("Total_Penjualan", axis=1)
    y = df["Total_Penjualan"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Model training complete with MLflow autologging.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RandomForestRegressor model.")
    parser.add_argument("--preprocessed-data-path", type=str, default="instax_sales_preprocessing.csv",
                        help="Path to the preprocessed data CSV file.")
    args = parser.parse_args()

    train_model(args.preprocessed_data_path)
