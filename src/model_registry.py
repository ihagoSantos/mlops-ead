import os
import mlflow
import numpy as np

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKNG_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD")

os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_PASSWORD
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Criando um client para comunicar com o registro DagsHub
client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

# Recebendo o modelo registrado e suas versões
registered_model = client.get_registered_model("fetal_health")
print("Latest version: ",registered_model.latest_versions)

# Obtendo o ID da execução do modelo
run_id = registered_model.latest_versions[-1].run_id
print("Run ID: ", run_id)

# Carregando o modelo
logged_model = f'runs:/{run_id}/model'
loaded_model = mlflow.pyfunc.load_model(logged_model)
print("Loaded Model: ", loaded_model)

# Fazendo predições com o modelo
accelerations = 0
fetal_movement = 0
uterine_constractions = 0
severe_declarations = 0

received_data = np.array([[accelerations, fetal_movement, uterine_constractions, severe_declarations]]).reshape(1, -1)
print ("Received Data: ", received_data)

predicted = loaded_model.predict(received_data)
print("Predicted: ", predicted)