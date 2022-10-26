import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import mlflow

from prefect import task, flow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc_fhv-orcherstration")

mlflow.autolog()

def read_data(path:str='./data/fhv_tripdata_2022-01.parquet'):
    df = pd.read_parquet(path)
    return df
    
    
def prepare_features(df, categorical, train=True):
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    mean_duration = df.duration.mean()
    if train:
        print(f"The mean duration of training is {mean_duration}")
    else:
        print(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

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

        print(f"The shape of X_train is {X_train.shape}")
        print(f"The DictVectorizer has {len(dv.feature_names_)} features")

        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_train)
        rmse = mean_squared_error(y_train, y_pred, squared=False)
        mlflow.log_metric("RMSE", rmse)
        print(f"The RMSE of training is: {rmse}")
    return lr, dv


def run_model(df, categorical, dv, lr):
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"The RMSE of validation is: {rmse}")
    return y_pred, y_val


def main(train_path: str = './data/fhv_tripdata_2022-01.parquet', 
           val_path: str = './data/fhv_tripdata_2022-02.parquet'):

    categorical = ['PUlocationID', 'DOlocationID']

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical)
    run_model(df_val_processed, categorical, dv, lr)


if __name__ == "__main__":
    main()

