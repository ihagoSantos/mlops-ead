import os
import random
import numpy as np
import random as python_random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from dataset import Dataset
from model import Model

from dotenv import load_dotenv

import mlflow
import dagshub

def reset_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":

    print("init")
    
    load_dotenv()

    data = Dataset(
            scaler = preprocessing.StandardScaler(),
            split = train_test_split,
            data_url = 'https://raw.githubusercontent.com/renansantosmendes/lectures-cdas-2023/master/fetal_health_reduced.csv'
    )

    data.set_train_test_data()

    print(data.X_train.head())
    print(data.X_test.head())
    print(data.y_train.head())
    print(data.y_test.head())

    model = Model(data.X_train)

    model.compile_model()
    
    # initializing dagshub
    #dagshub.init(
    #    repo_owner='ihagosantos',
    #    repo_name='mlops-ead',
    #    mlflow=True
    #)

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

    mlflow.tensorflow.autolog(
        log_models=True,
        log_input_examples=True,
    )

    with mlflow.start_run( run_name='experiment_0mlops_ead' ) as run:

        model.train_model(
            data.X_train,
            data.y_train
        )



