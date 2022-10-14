import argparse
import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def run(data_path):

    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

    os.environ["AWS_PROFILE"] = "soumyadip"
    TRACKING_SERVER_HOST = "ec2-xxxxxxxx.xxxxxxx.compute.amazonaws.com"
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
    mlflow.set_experiment("NYC-trip-duration-prediction")
    mlflow.autolog()
    with mlflow.start_run():
        mlflow.set_tags({
            "developer": "soumyadip",
            "problem": "nyc-taxi",
            "model-type": "Prediction"
        })
        mlflow.log_params({
            "train-data-path": "./data/green_tripdata_2021-01.parquet",
            "valid-data-path": "./data/green_tripdata_2021-02.parquet",
            "test-data-path": "./data/green_tripdata_2021-03.parquet",
        })
        mlflow.log_artifact("output/dv.pkl", artifact_path="preprocessor")

        model = RandomForestRegressor(max_depth=10, random_state=143)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_valid)

        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="location of NYC data"
    )
    args = parser.parse_args()

    run(args.data_path)