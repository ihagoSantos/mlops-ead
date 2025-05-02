import os
import json
import mlflow
import uvicorn
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI
from contextlib import asynccontextmanager
from fastapi import HTTPException

class FetalHealthDTO(BaseModel):
    accelerations: float
    fetal_movement: float
    uterine_contractions: float
    severe_decelerations: float

app = FastAPI(
    title="Fetal Health API",
    openapi_tags=[
        { "name":"Health", "description":"GET API Health" },
        { "name":"Prediction", "description":"Model Prediction" },
    ],
)
@app.on_event(event_type='startup')
def startup_event():
    """
    A function that is called when the application starts up. It loads a model into the
    global variable `model`.

    Parameters:
        None

    Returns:
        None
    """
    global model
    model = load_model()

def load_model():
    try:
        print('reading model...')
        MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
        MLFLOW_TRACKING_USERNAME = os.environ.get("MLFLOW_TRACKING_USERNAME")
        MLFLOW_TRACKING_PASSWORD = os.environ.get("MLFLOW_TRACKING_PASSWORD")

        os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
        print('setting mlflow...')
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        print('creating client...')
        client = mlflow.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)  
        print('registering model...')
        registered_model = client.get_registered_model('fetal_health')
        print('loading model...')
        run_id = registered_model.latest_versions[-1].run_id
        loaded_model = mlflow.pyfunc.load_model(f'runs:/{run_id}/model')
        print('model:', loaded_model)
        return loaded_model
    except Exception as e:
        print(e)


@app.get(path="/healthy", tags=["Health"])
def healthy():
    return {"status": "healthy"}

@app.post(path="/predict", tags=["Prediction"])
def predict(request: FetalHealthDTO):
    try:
        print("request:", request)
        
        global model
        print(model)
        if model is None:
            raise HTTPException(status_code=422, detail="Modelo não foi carregado.")
        
        received_data = np.array([
            request.accelerations,
            request.fetal_movement,
            request.uterine_contractions,
            request.severe_decelerations
        ]).reshape(1, -1)

        print("received data:", received_data)
        
        prediction = model.predict(received_data)
        print("prediction:", prediction)
        max_index = np.argmax(prediction[0])
        return {
            "prediction": {
                "class": str(max_index),
                "probability": str(prediction[0][max_index])
            }   
        }
    except Exception as e:
        print("Erro:", str(e))
        raise HTTPException(status_code=422, detail="Erro ao realizar predição dos dados.") 