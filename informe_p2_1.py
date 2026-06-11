import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm #barra de progreso
# --------------------------------PARTE 2-------------------------------

# from sklearn.model_selection import train_test_split

# Cargar dataset
df = pd.read_csv('OnlineRetail.csv', sep=';', encoding='latin1')
df.columns = df.columns.str.replace('ï»¿', '').str.strip()

# Corregir UnitPrice (coma decimal)
df['UnitPrice'] = df['UnitPrice'].astype(str).str.replace(',', '.').astype(float)

# Limpiar datos
df = df[df['Quantity'] > 0]                                      # eliminar devoluciones
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]        # eliminar cancelaciones
df = df[df['UnitPrice'] > 0]                                     # eliminar precio 0
df = df[df['Quantity'] <= 1000]                                  # eliminar outliers extremos
df = df.dropna(subset=['UnitPrice'])                             # eliminar nulos

# Crear variable Country binaria
df['CountryBinario'] = (df['Country'] == 'United Kingdom').astype(int)

# Crear etiqueta
# df['Salida'] = (df['Quantity'] > 25).astype(int)

#NUEVO QUE AGREGO
# frecuencia = df.groupby('CustomerID')['InvoiceNo'].nunique()
# df['Frecuencia'] = df['CustomerID'].map(frecuencia).fillna(1)
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
# df['Salida'] = ((df['Quantity'] > 25) & (df['Frecuencia'] > 1)).astype(int)
df['Salida'] = ((df['Quantity'] > 20) & (df['TotalPrice'] > 30)).astype(int)

# Normalización Z-score
# for col in ['Quantity', 'UnitPrice']:
# for col in ['Quantity', 'UnitPrice', 'Frecuencia']:
for col in ['Quantity', 'UnitPrice', 'TotalPrice']:
    media = df[col].mean()
    desvio = df[col].std()
    df[col] = (df[col] - media) / desvio

# Definir salida
X = df[['Quantity', 'UnitPrice', 'CountryBinario', 'TotalPrice']].values
Y = df['Salida'].values

# Dividir en entrenamiento y test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)
n = X_train.shape[0]

# Inicializar pesos y sesgos aleatoriamente
w_hidden = np.random.rand(4, 4)  # 4 neuronas ocultas, 3 inputs
w_output = np.random.rand(1, 4)  # 1 salida, 4 neuronas ocultas

b_hidden = np.random.rand(4, 1)  # 1 sesgo por neurona oculta
b_output = np.random.rand(1, 1)  # 1 sesgo para la salida

# Funciones de activación
relu = lambda x: np.maximum(x, 0)
logistic = lambda x: 1 / (1 + np.exp(-x))

# Propagación hacia adelante
def forward_prop(X):
    Z1 = w_hidden @ X + b_hidden
    A1 = relu(Z1)
    Z2 = w_output @ A1 + b_output
    A2 = logistic(Z2)
    return Z1, A1, Z2, A2

def costo(A2, Y):
    return np.mean((A2 - Y)**2)

# Derivadas de las funciones de activación
d_relu = lambda x: x > 0
d_logistic = lambda x: np.exp(-x) / (1 + np.exp(-x)) ** 2

# Backpropagation
def backward_prop(Z1, A1, Z2, A2, X, Y):
    dC_dA2 = 2 * A2 - 2 * Y
    dA2_dZ2 = d_logistic(Z2)
    dZ2_dA1 = w_output
    dZ2_dW2 = A1
    dZ2_dB2 = 1
    dA1_dZ1 = d_relu(Z1)
    dZ1_dW1 = X
    dZ1_dB1 = 1

    dC_dW2 = dC_dA2 @ dA2_dZ2 @ dZ2_dW2.T
    dC_dB2 = dC_dA2 @ dA2_dZ2 * dZ2_dB2
    dC_dA1 = dC_dA2 @ dA2_dZ2 @ dZ2_dA1
    dC_dW1 = dC_dA1 @ dA1_dZ1 @ dZ1_dW1.T
    dC_dB1 = dC_dA1 @ dA1_dZ1 * dZ1_dB1

    return dC_dW1, dC_dB1, dC_dW2, dC_dB2

L = 0.05 # Es mi tasa de aprendizaje. Si es muy grande son muy bruscos y muy chicos no aprende

# Listas para guardar métricas
costos_train = []
costos_test = []
accuracies_train = []
accuracies_test = []

for i in tqdm(range(100_000)):
    # Seleccionar aleatoriamente un dato de entrenamiento
    idx = np.random.choice(n, 1, replace=False)
    X_sample = X_train[idx].transpose()
    Y_sample = Y_train[idx]

    # Forward propagation
    Z1, A1, Z2, A2 = forward_prop(X_sample)

    # Backpropagation
    dW1, dB1, dW2, dB2 = backward_prop(Z1, A1, Z2, A2, X_sample, Y_sample)

    # Actualizar pesos y sesgos
    w_hidden -= L * dW1
    b_hidden -= L * dB1
    w_output -= L * dW2
    b_output -= L * dB2

    # Guardar métricas cada 1000 iteraciones
    if i % 1000 == 0:
        _, _, _, A2_train = forward_prop(X_train.T)
        _, _, _, A2_test = forward_prop(X_test.T)
        costos_train.append(costo(A2_train, Y_train))
        costos_test.append(costo(A2_test, Y_test))
        acc_train = np.mean((A2_train >= 0.5) == Y_train)
        acc_test = np.mean((A2_test >= 0.5) == Y_test)
        accuracies_train.append(acc_train)
        accuracies_test.append(acc_test)

# Grafico
plt.figure(figsize=(12, 5)) #tamaño de mi cuadro, 12x5

plt.subplot(1, 2, 1) #divide el tamaño en 1 fila y 2 columnas, y toma el primero
plt.plot(costos_train, label='Entrenamiento') #dibuja la línea de entrenamiento
plt.plot(costos_test, label='Test') #dibuja la línea de tests
plt.title('Función de pérdida') #es el título del gráfico
plt.xlabel('Iteraciones (x1000)') #título del eje X
plt.ylabel('Costo') #título del eje Y
plt.legend() #muestra la leyenda

plt.subplot(1, 2, 2) #ahora tomo el segundo gráfico
plt.plot(accuracies_train, label='Entrenamiento')
plt.plot(accuracies_test, label='Test')
plt.title('Accuracy') #título
plt.xlabel('Iteraciones (x1000)') 
plt.ylabel('Precisión')
plt.legend()

plt.tight_layout() #ajusta para que no se pisen los gráficos
plt.show() #muestro la ventana

_, _, _, A2_test = forward_prop(X_test.T)
predicciones = (A2_test >= 0.5).astype(int).flatten()

mayoristas_reales = Y_test.sum()
mayoristas_predichos = predicciones.sum()

print(f'Mayoristas reales en test: {mayoristas_reales}')
print(f'Mayoristas predichos: {mayoristas_predichos}')

_, _, _, A2_train = forward_prop(X_train.T)
_, _, _, A2_test = forward_prop(X_test.T)

acc_train = np.mean((A2_train >= 0.5) == Y_train)
acc_test = np.mean((A2_test >= 0.5) == Y_test)

print("Puntaje entrenamiento: %f" % acc_train)
print("Puntaje test: %f" % acc_test)

print(f'Detección de mayoristas: {mayoristas_predichos/mayoristas_reales*100:.1f}%')