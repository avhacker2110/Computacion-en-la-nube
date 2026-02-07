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

def test_predict_survival_female_first_class():
    """Test predicción: mujer joven de primera clase (probablemente sobrevive)"""
    body = {
        "edad": 12,
        "clase": "First",
        "sexo": "f"  # ✅ minúscula
    }
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    assert "nivel de salades" in response.json()
    print(f"Predicción: {response.json()}") 


def test_predict_survival_male_first_class():
    """Test predicción: mujer de primera clase (probablemente sobrevive)"""
    body = {
        "edad": 12,
        "clase": "First",
        "sexo": "m"  # ✅ minúscula
    }
    response = client.post("/predict", json=body)
    assert response.status_code == 200
    assert "nivel de salades" in response.json()
    print(f"Predicción: {response.json()}") 


def test_predict_survival_child():
    """Test predicción: niño"""
    body = {
        "edad": 7,
        "clase": "Second",
        "sexo": "m"  # ✅ minúscula
    }
    response = client.post("/predict", json=body)
    print(f"\nNiño: {response.json()}")
    assert response.status_code == 200
    assert "nivel de salades" in response.json()

def test_predict_survival_childf():
    """Test predicción: niña"""
    body = {
        "edad": 7,
        "clase": "Second",
        "sexo": "f"  # ✅ minúscula
    }
    response = client.post("/predict", json=body)
    print(f"\nNiña: {response.json()}")
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

def test_predict_survival_young_man():
    """Test adicional: hombre joven"""
    body = {
        "edad": 25,
        "clase": "Second",
        "sexo": "m"
    }
    response = client.post("/predict", json=body)
    print(f"\nHombre joven: {response.json()}")
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
    assert "nivel de salades" in response.json()

def test_predict_survival_old_woman():
    """Test adicional: mujer mayor"""
    body = {
        "edad": 60,
        "clase": "Third",
        "sexo": "f"
    }
    response = client.post("/predict", json=body)
    print(f"\nMujer mayor: {response.json()}")
    assert response.status_code == 200
    assert "nivel de salades" in response.json()