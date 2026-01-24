from fastapi import FastAPI
from pydantic import BaseModel  
from uvicorn import run
import numpy as np
import pandas as pd


app = FastAPI()
class Prediction(BaseModel):
    edad: int
    clase: str
    sexo: str

@app.get("/test/")# Endpoint de prueba
def testapi():
    return {"message": "API is working"}