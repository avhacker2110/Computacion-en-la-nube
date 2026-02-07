import pytest
from fastapi.testclient import TestClient
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import app

client = TestClient(app)

def test_api_running():
    """Test para verificar que la API está funcionando"""
    response = client.get("/test/")  # ✅ Agregado /
    assert response.status_code == 200
    # La respuesta es una lista, no un dict
    assert response.json() == ["API de predicción de supervivencia del Titanic"]

def test_predict_survival_male_third_class():
    """Test predicción: hombre joven de tercera clase (probablemente no sobrevive)"""
    body = {
        "edad": 22,
        "clase": "Third",
        "sexo": "m"  # ✅ minúscula
    }
    response = client.post("/predict", json=body)
    print(f"\nHombre tercera clase: {response.json()}")
    assert response.status_code == 200
    assert "nivel de salades" in response.json()
    assert response.json()["nivel de salades"] == "chulo con papas"

def test_predict_survival_female_first_class():
    """Test predicción: mujer de primera clase (probablemente sobrevive)"""
    body = {
        "edad": 35,
        "clase": "First",
        "sexo": "f"  # ✅ minúscula
    }
    response = client.post("/predict", json=body)
    print(f"\nMujer primera clase: {response.json()}")
    assert response.status_code == 200
    data = response.json()
    assert "nivel de salades" in data
    # ✅ Mensaje actualizado
    assert data["nivel de salades"] == "boleto pal proximo titanic"

def test_predict_survival_child():
    """Test predicción: niño"""
    body = {
        "edad": 5,
        "clase": "Second",
        "sexo": "m"  # ✅ minúscula
    }
    response = client.post("/predict", json=body)
    print(f"\nNiño: {response.json()}")
    assert response.status_code == 200
    assert "nivel de salades" in response.json()

def test_predict_survival_young_woman():
    """Test adicional: mujer joven"""
    body = {
        "edad": 25,
        "clase": "Second",
        "sexo": "f"
    }
    response = client.post("/predict", json=body)
    print(f"\nMujer joven: {response.json()}")
    assert response.status_code == 200
    assert "nivel de salades" in response.json()

def test_predict_survival_old_man():
    """Test adicional: hombre mayor"""
    body = {
        "edad": 60,
        "clase": "Third",
        "sexo": "m"
    }
    response = client.post("/predict", json=body)
    print(f"\nHombre mayor: {response.json()}")
    assert response.status_code == 200
    data = response.json()
    assert "nivel de salades" in data
    assert data["nivel de salades"] == "chulo con papas"