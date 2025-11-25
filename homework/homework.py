#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#


import pandas as pd
import os
import gzip
import pickle
import json
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error

# Load and preprocess data
def load_and_preprocess(path):
    df = pd.read_csv(path, compression="zip")
    df['Age'] = 2021 - df['Year']
    df.drop(['Year', 'Car_Name'], axis=1, inplace=True)
    return df.drop('Present_Price', axis=1), df['Present_Price']

x_train, y_train = load_and_preprocess("files/input/train_data.csv.zip")
x_test, y_test = load_and_preprocess("files/input/test_data.csv.zip")

# Create and train pipeline
cat_cols = ['Fuel_Type', 'Selling_type', 'Transmission']
num_cols = ['Selling_Price', 'Driven_kms', 'Owner', 'Age']

model = GridSearchCV(
    Pipeline([
        ('preprocessor', ColumnTransformer([
            ('cat', OneHotEncoder(), cat_cols),
            ('num', MinMaxScaler(), num_cols)
        ])),
        ('feature_selection', SelectKBest(f_regression)),
        ('regressor', LinearRegression())
    ]),
    {'feature_selection__k': range(1, 12)},
    cv=10,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
).fit(x_train, y_train)

# Save model
os.makedirs('files/models', exist_ok=True)
with gzip.open('files/models/model.pkl.gz', 'wb') as f:
    pickle.dump(model, f)

# Save metrics
os.makedirs('files/output', exist_ok=True)
with open('files/output/metrics.json', 'w') as f:
    for name, X, y in [('train', x_train, y_train), ('test', x_test, y_test)]:
        pred = model.predict(X)
        f.write(json.dumps({
            'type': 'metrics',
            'dataset': name,
            'r2': float(r2_score(y, pred)),
            'mse': float(mean_squared_error(y, pred)),
            'mad': float(median_absolute_error(y, pred))
        }) + '\n')