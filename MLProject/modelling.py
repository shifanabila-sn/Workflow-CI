import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import mlflow
import mlflow.sklearn
import os
import argparse 

# Enable MLflow autologging for scikit-learn
mlflow.sklearn.autolog()

def train_model(preprocessed_data_path):
    print(f"Current working directory: {os.getcwd()}")
    print(f"Loading preprocessed data from: {preprocessed_data_path}")

    # Muat dataset dari path yang diberikan sebagai argumen
    # Pastikan file 'instax_sales_preprocessing.csv' ada di folder MLProject
    df = pd.read_csv(preprocessed_data_path)

    # Definisikan fitur (X) dan target (y)
    X = df.drop("Total_Penjualan", axis=1)
    y = df["Total_Penjualan"]

    # Bagi dataset menjadi data pelatihan dan data pengujian
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inisialisasi dan latih model RandomForestRegressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Lakukan prediksi pada data pengujian (autologging akan mencatat metrik secara otomatis)
    y_pred = model.predict(X_test)

    print("Model training complete with MLflow autologging.")
    # Hapus atau komentar baris ini karena tidak relevan di GitHub Actions
    # print("Check MLflow UI for experiment results: http://localhost:5000")

if __name__ == "__main__":
    # Tambahkan parser argumen untuk menerima path data
    parser = argparse.ArgumentParser(description="Train RandomForestRegressor model.")
    parser.add_argument("--preprocessed-data-path", type=str, default="instax_sales_preprocessing.csv",
                        help="Path to the preprocessed data CSV file.")
    args = parser.parse_args()

    train_model(args.preprocessed_data_path)
