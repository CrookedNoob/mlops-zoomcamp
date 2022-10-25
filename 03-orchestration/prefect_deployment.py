
from multiprocessing.resource_sharer import stop
import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule
from prefect.filesystems import S3
from prefect.blocks.core import Block


# from datetime import timedelta

# s3_block = S3.load("nyc-trip-duration-model")
# storage = Block.load("s3/nyc-trip-duration-model")



mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc_experiment-orcherstration")


def read_dataframe(filename):
    """ Read parquet files as pandas dataframe
    change the duration into proper datetime format
    """
    
    df = pd.read_parquet(filename)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

@task
def add_features(train_path, val_path):

    """Feature engineering
    vectorize labelled dictionary into array
    """

    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)

    print(f"No. of obs in Train set: {len(df_train)}")
    print(f"No. of obs in Valid set: {len(df_val)}")

    df_train["PU_DO"] = df_train["PULocationID"] + "_" + df_train["DOLocationID"]
    df_val["PU_DO"] = df_val["PULocationID"] + "_" + df_val["DOLocationID"]

    categorical = ["PU_DO"]
    numerical  = ["trip_distance"]

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient="records")
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient="records")
    X_val = dv.transform(val_dicts)

    target = 'duration'
    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv

@task
def train_model_search(train, valid, y_val):
    """Search for the best opyomized hyperparameters for model"""
    
    def _objective(params):
        """train model to evaluate the performance"""
        
        with mlflow.start_run():
            mlflow.set_tags({
                "model": "xgboost",
                "data_scientist": "soumyadip",
                "type": "training" 
            })
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=500,
                evals=[(valid, "validation")],
                early_stopping_rounds=50 
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)

            return {"loss": rmse, "status": STATUS_OK}


    search_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 4, 100, 1)),
        "learning_rate": hp.loguniform("learning_rate", -3, 0),
        "reg_alpha": hp.loguniform("reg_alpha", -5, -1),
        "reg_lambda": hp.loguniform("reg_lambda", -6, -1),
        "min_child_weight": hp.loguniform("min_child_weight", -1, 3),
        "objective": "reg:squarederror",
        "random_state": 142
    }

    best_result = fmin(
        fn=_objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1,
        trials=Trials()
    )

    return best_result

@task
def train_best_model(train, valid, y_val, dv, best_params):
    
    with mlflow.start_run():
        mlflow.set_tags({
            "model": "xgboost",
            "data_scientist": "soumyadip",
            "type": "final" 
            })
        #train = xgb.DMatrix(X_train, label=y_train)
        #valid = xgb.DMatrix(X_valid, label=y_val)

        best_params = best_params

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=500,
            evals=[(valid, "validation")],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)
        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
        
        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")

@flow(task_runner=SequentialTaskRunner)
def main(train_path: str ="./data/green_tripdata_2021-01.parquet",
        val_path: str = "./data/green_tripdata_2021-02.parquet"):
    X_train, X_val, y_train, y_val, dv = add_features(train_path, val_path)
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    best_params = train_model_search(train, valid, y_val)
    best_params['max_depth']= int(best_params['max_depth'])
    print(best_params) 
    train_best_model(train, valid, y_val, dv, best_params)
    

if __name__ == "__main__":
    main()

    # deployment = Deployment.build_from_flow(
    #     flow=main,
    #     name="model_training",
    #     schedule=IntervalSchedule(interval=timedelta(minutes=5)),
    #     work_queue_name="ml",
    #     #storage=storage
    # )
    
    # deployment.apply()
#    main()
