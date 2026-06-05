import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Cargar dataset
df = pd.read_csv('OnlineRetail.csv', sep=';', encoding='latin1')
df.columns = df.columns.str.replace('ï»¿', '').str.strip()

# Corregir UnitPrice
df['UnitPrice'] = df['UnitPrice'].astype(str).str.replace(',', '.').astype(float)

# Limpiar datos
df = df[df['Quantity'] > 0]
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[df['UnitPrice'] > 0]
df = df[df['Quantity'] <= 1000]
df = df.dropna(subset=['UnitPrice'])

# Crear variables
df['CountryBinario'] = (df['Country'] == 'United Kingdom').astype(int)
df['Salida'] = (df['Quantity'] > 25).astype(int)

# Normalización Z-score
for col in ['Quantity', 'UnitPrice']:
    media = df[col].mean()
    desvio = df[col].std()
    df[col] = (df[col] - media) / desvio

# Features y salida
X = df[['Quantity', 'UnitPrice', 'CountryBinario']].values
Y = df['Salida'].values

# Dividir en entrenamiento y test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)

# Red neuronal con sklearn
nn = MLPClassifier(solver='sgd',
                   hidden_layer_sizes=(4, ),
                   activation='relu',
                   max_iter=100_000,
                   learning_rate_init=0.05)

nn.fit(X_train, Y_train)

print("Puntaje entrenamiento: %f" % nn.score(X_train, Y_train))
print("Puntaje test: %f" % nn.score(X_test, Y_test))