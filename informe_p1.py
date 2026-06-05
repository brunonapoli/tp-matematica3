import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv('OnlineRetail.csv', delimiter=';')
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce') #convierto la columna en números reales y en caso que haya texto u otra cosa, lo tome como NaN
#convierto en texto para cambiar las COMAS por PUNTOS y ahí tener valores con punto
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'].astype(str).str.replace(',', '.'), errors='coerce')
#borro todos los NaN que me detectaron en los pasos anteriores
df = df.dropna(subset=['Quantity', 'UnitPrice'])

#con el .astyp(int) convierto el TRUE en 1 y el FALSE en 0
df['Mayorista'] = (df['Quantity'] > 25).astype(int)
df['Valor total'] = df['Quantity'] * df['UnitPrice']
#convierto en 1 si es de United Kingdom
df['United Kingdom'] = (df['Country'] == 'United Kingdom').astype(int)

caracteristicas = ['Quantity', 'UnitPrice', 'Valor total', 'United Kingdom', 'Mayorista']
carac_correlacion = df[caracteristicas].corr()
# print(carac_correlacion['Mayorista'].sort_values(ascending = False))

print(df['Quantity'].max()) #veo el valor máximo, acá me tira 2880
print(df['Quantity'].sort_values(ascending=False).head(10)) #veo los 10 valores mas altos
print(df['Quantity'].mean())   # promedio
print(df['Quantity'].median()) # mediana

# Para calcular la normalización:
for columna in ['Quantity', 'UnitPrice']:
    media = df[columna].mean()
    desvio = df[columna].std()
    df[columna] = (df[columna] - media) / desvio
