import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import os

# Enable MLflow autologging for scikit-learn
mlflow.sklearn.autolog()

def train_model():
    print(f"Current working directory: {os.getcwd()}")

    # Muat dataset (sudah disalin ke Membangun_model/)
    df = pd.read_csv("instax_sales_preprocessing.csv")

    X = df.drop("Total_Penjualan", axis=1)
    y = df["Total_Penjualan"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("Model training complete with MLflow autologging.")
    print("Check MLflow UI for experiment results: http://localhost:5000")

if __name__ == "__main__":
    train_model()
