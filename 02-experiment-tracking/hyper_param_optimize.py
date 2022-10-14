import argparse
from ast import arg
import os
import pickle
import numpy as np

import mlflow
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

os.environ["AWS_PROFILE"] = "soumyadip"
TRACKING_SERVER_HOST = "ec2-xxxxxxx.xxxxxxxx.compute.amazonaws.com"
mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")
mlflow.set_experiment("NYC-trip-duration-prediction")

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def run(data_path, num_trials):


    
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

    def objective(params):
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
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_valid)

            rmse = mean_squared_error(y_valid, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

        return {"loss": rmse, "status": STATUS_OK}

    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 1, 20, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 10, 50, 1)),
        "min_samples_split": scope.int(hp.quniform("min_samples_split", 2, 10, 1)),
        "min_samples_leaf": scope.int(hp.quniform("min_samples_leaf", 1, 5, 1)),
        "random_state": 143
    }

    rstate = np.random.default_rng(143)
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="location of NYC data"
    )
    parser.add_argument(
        "--max_evals",
        type=int,
        default=50,
        help="number of parameter evaluations for the optimizer to explore"
    )

    args = parser.parse_args()

    run(args.data_path, args.max_evals)