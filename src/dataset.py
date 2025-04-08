import pandas as pd

class Dataset:
    def __init__(self, scaler, split, data_url, test_size=0.3, random_state=42):
        self.scaler = scaler
        self.split = split
        self.data_url = data_url
        self.test_size = test_size
        self.random_state = random_state
        self.load_data()
        
    def load_data(self):
        if(self.data_url == None):
            raise Exception("data_url must be a string")

        self.data = pd.read_csv(self.data_url)
        self.X = self.data.drop(["fetal_health"], axis=1)
        self.y = self.data["fetal_health"]
        self.columns_names = list(self.X)

    def set_train_test_data(self): 
        X_df = self.scaler.fit_transform(self.X)
        X_df = pd.DataFrame(X_df, columns=self.columns_names)

        self.X_train, self.X_test, self.y_train, self.y_test = self.split(X_df, self.y, test_size = self.test_size, random_state=self.random_state)

        self.y_train = self.y_train - 1
        self.y_test = self.y_test - 1

        
        
