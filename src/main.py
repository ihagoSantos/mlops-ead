import os
import random
import numpy as np
import random as python_random
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from dataset import Dataset
from model import Model

def reset_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    print("init")

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

    model.train_model(
        data.X_train,
        data.y_train
    )
