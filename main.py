from fastapi import FastAPI
from pydantic import BaseModel  
from uvicorn import run
import numpy as np
import pandas as pd
import pickle


with open('modelo_titanic.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()
class Prediction(BaseModel):
    edad: int
    clase: str
    sexo: str

@app.get("/test/")# Endpoint de prueba
def testapi():
    return {"API de predicción de supervivencia del Titanic"}

@app.post("/predict")
def predict_survival(data: Prediction):
    # Convertir datos de entrada a DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Preprocesamiento 
    input_data['sexo'] = input_data['sexo'].map({'m': 0, 'f': 1})
    input_data['clase'] = input_data['clase'].map({'First': 1, 'Second': 2, 'Third': 3})
    input_data = input_data.fillna(input_data.mean())
    
    # Seleccionar características
    features = input_data[['edad', 'clase', 'sexo']]
    
    # Realizar predicción
    prediction = model.predict(features)
    
    # Interpretar resultado
    survival = 'boleto pal proximo titanic' if prediction[0] == 1 else 'chulo con papas'
    
    return {"nivel de salades": survival}