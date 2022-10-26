import pandas as pd
import pickle
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import task, flow, get_run_logger
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule

import mlflow


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc_fhv-orcherstration")

mlflow.autolog()

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df
    
@task   
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    mean_duration = df.duration.mean()
    logger = get_run_logger()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):

    with mlflow.start_run():
        mlflow.set_tags({
            "model": "Linear Regression",
            "data_scientist": "soumyadip",
            "type": "dev" 
        })
        mlflow.set_tag("developer", "soumyadip")

        train_dicts = df[categorical].to_dict(orient='records')
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts) 
        y_train = df.duration.values
        logger = get_run_logger()
        logger.info(f"The shape of X_train is {X_train.shape}")
        logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_train)
        rmse = mean_squared_error(y_train, y_pred, squared=False)
        mlflow.log_metric("RMSE", rmse)
        logger.info(f"The RMSE of training is: {rmse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    logger = get_run_logger()
    logger.info(f"The RMSE of validation is: {rmse}")
    return y_pred, y_val

@task
def get_paths(date):
    if date:
        processed_date = datetime.strptime(date, "%Y-%m-%d")
    else:
        processed_date = datetime.today()
    logger = get_run_logger()
    logger.info(f"Processing train date")
    train_date = processed_date - relativedelta(months=2)
    logger.info(f"Processing val date")
    val_date = processed_date - relativedelta(months=1)
    logger.info(f"Reading train data")
    train_path = f"./data/fhv_tripdata_2021-{str(train_date.month).zfill(2)}.parquet"
    logger.info(f"Reading val data")
    val_path = f"./data/fhv_tripdata_2021-{str(val_date.month).zfill(2)}.parquet"
    return train_path, val_path


@flow(task_runner=SequentialTaskRunner)
def main(date=None):

    train_path, val_path = get_paths(date)
    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical)
    run_model(df_val_processed, categorical, dv, lr)

    if date is None:
        date = datetime.today.strftime("%Y-%m-%d")

    logger = get_run_logger()
    logger.info(f"Storing Dict Vectorizer model in binary format")
    with open(f"./models/dv-{date}.b" , "wb") as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")


deployment = Deployment.build_from_flow(
        flow=main,
        name="fhv_model_train-homework-interval",
        schedule=IntervalSchedule(interval=timedelta(minutes=5)),
        work_queue_name="fhv-interval-ml",
    )

if __name__ == "__main__":
    #main("2021-10-15")
    deployment.apply()

#   prefect agent start --work-queue fhv-interval-ml