import argparse
from http import client
import os
import pickle

import mlflow
from hyperopt import hp, space_eval
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMAENT_NAME = "NYC-trip-duration-prediction"
EXPERIMENT_NAME = "random-forest-best-models"

TRACKING_SERVER_HOST = "ec2-xxxxxxxx.xxxxx.compute.amazonaws.com"
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

mlflow.set_experiment(EXPERIMENT_NAME)

mlflow.autolog()

SPACE = {
    "max_depth": scope.int(hp.quniform("max_depth", 1, 20, 1)),
    "n_estimators": scope.int(hp.quniform("n_estimators", 10, 50, 1)),
    "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
    "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 4, 1)),
    "random_state": 143
}


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

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

        params = space_eval(SPACE, params)
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        valid_rmse = mean_squared_error(y_valid, rf.predict(X_valid), squared=False)
        mlflow.log_metric("valid_rmse", valid_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)

def run(data_path, log_top):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(HPO_EXPERIMAENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=log_top,
        order_by=["metrics.rmse ASC"]
    )

    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"])[0]

    mlflow.register_model(
        model_uri=f"runs:/{best_run.info.run_id}/model",
        name = "nyc-taxi-duration"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="location of NYC data"
    )
    parser.add_argument(
        "--top_n",
        default=5, 
        type=int,
        help="the top 'top_n' models will be evaluated to decide which model to promote"
    )

    args = parser.parse_args()
    
    run(args.data_path, args.top_n)