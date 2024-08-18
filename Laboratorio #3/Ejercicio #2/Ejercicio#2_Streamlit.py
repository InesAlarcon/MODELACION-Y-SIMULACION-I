import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Configuración de la aplicación
st.title('Simulación de Publicidad para Maximizar Ventas')

# Cargar los datos
@st.cache_data
def cargar_datos():
    return pd.read_csv('Advertising.csv')

df = cargar_datos()

# Mostrar el resumen de los datos
st.subheader('Resumen de los Datos')
st.write(df.describe())
st.write(df.corr())

# Separar variables explicatorias y variable objetivo
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y ajustar el modelo
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = modelo.predict(X_test)

# Calcular el error cuadrático medio
mse = mean_squared_error(y_test, y_pred)

st.subheader('Resultados del Modelo')
st.write(f'Error cuadrático medio: {mse:.2f}')
st.write('Coeficientes del modelo:', modelo.coef_)
st.write('Intercepto:', modelo.intercept_)

# Función para obtener los parámetros de la distribución triangular
def obtener_parametros_distribucion(variable):
    min_val = variable.min()
    max_val = variable.max()
    mode_val = variable.median()

    if mode_val < min_val:
        mode_val = min_val
    elif mode_val > max_val:
        mode_val = max_val

    return min_val, mode_val, max_val

# Obtener parámetros para TV, Radio y Newspaper
tv_params = obtener_parametros_distribucion(df['TV'])
radio_params = obtener_parametros_distribucion(df['Radio'])
newspaper_params = obtener_parametros_distribucion(df['Newspaper'])

st.subheader('Parámetros de Distribución')
st.write(f'Parámetros para TV: {tv_params}')
st.write(f'Parámetros para Radio: {radio_params}')
st.write(f'Parámetros para Newspaper: {newspaper_params}')

# Función para la simulación de Montecarlo
def simulacion_montecarlo(tv_params, radio_params, newspaper_params, intentos=1000):
    resultados = []
    for _ in range(intentos):
        tv = np.random.triangular(*tv_params)
        radio = np.random.triangular(*radio_params)
        newspaper = np.random.triangular(*newspaper_params)
        entrada = pd.DataFrame([[tv, radio, newspaper]], columns=['TV', 'Radio', 'Newspaper'])
        ventas = modelo.predict(entrada)[0]
        resultados.append((tv, radio, newspaper, ventas))
    return resultados

# Realizar la simulación de Montecarlo
resultados = simulacion_montecarlo(tv_params, radio_params, newspaper_params)
resultados_df = pd.DataFrame(resultados, columns=['TV', 'Radio', 'Newspaper', 'Sales'])

st.subheader('Resultados de la Simulación de Montecarlo')
st.write(resultados_df.head())

# Cálculo de los promedios y normalización
promedios = resultados_df.mean()
total_promedios = promedios['TV'] + promedios['Radio'] + promedios['Newspaper']
tv_percent = promedios['TV'] / total_promedios
radio_percent = promedios['Radio'] / total_promedios
newspaper_percent = promedios['Newspaper'] / total_promedios

st.subheader('Presupuesto Normalizado')
st.write(f'Presupuesto normalizado (TV): {tv_percent:.2%}')
st.write(f'Presupuesto normalizado (Radio): {radio_percent:.2%}')
st.write(f'Presupuesto normalizado (Newspaper): {newspaper_percent:.2%}')
