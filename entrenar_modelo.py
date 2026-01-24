import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_openml

# Cargar dataset del Titanic
titanic = fetch_openml('titanic', version=1, as_frame=True, parser='auto')
df = titanic.data

# Preprocesamiento
df['survived'] = titanic.target.astype(int)
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['pclass'] = df['pclass'].astype(int)
df = df[['age', 'pclass', 'sex', 'survived']].dropna()

# Renombrar columnas para que coincidan con tu API
df.columns = ['edad', 'clase', 'sexo', 'survived']

# Separar características y objetivo
X = df[['edad', 'clase', 'sexo']]
y = df['survived']

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar modelo
with open('modelo_titanic.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Modelo entrenado y guardado como 'modelo_titanic.pkl'")
print(f"📊 Precisión en test: {model.score(X_test, y_test):.2%}")