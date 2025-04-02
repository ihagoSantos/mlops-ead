import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("Init script")
data = load_iris()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state=42
)

# init a experiment with mlflow
mlflow.start_run()

# create a Logistic Regression model
model = LogisticRegression(max_iter=1000)
# train data
model.fit(X_train, y_train)

# predict data
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# model metrics recorded 
mlflow.log_param("max_iter", model.max_iter)

# performance metrics recorded
mlflow.log_metric("accuracy", accuracy)

mlflow.sklearn.log_model(model, "model")

# end experiment
mlflow.end_run()


