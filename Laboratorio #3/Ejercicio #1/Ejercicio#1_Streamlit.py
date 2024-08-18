import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Parámetros del problema
ahorro_anual = 20000
años = 17
tasa_promedio = 0.04
desviacion_estandar = 0.10
num_simulaciones = 1000

# Función de simulación de Montecarlo
def simulacion_montecarlo(ahorro_anual, años, tasa_promedio, desviacion_estandar, num_simulaciones):
    resultados = []
    for _ in range(num_simulaciones):
        ahorros = 0
        rendimiento_anual = np.random.normal(tasa_promedio, desviacion_estandar, años)
        monto_acumulado = []
        for rendimiento in rendimiento_anual:
            ahorros = (ahorros + ahorro_anual) * (1 + rendimiento)
            monto_acumulado.append(ahorros)
        resultados.append(monto_acumulado)
    return np.array(resultados)

# Ejecución de la simulación
resultados = simulacion_montecarlo(ahorro_anual, años, tasa_promedio, desviacion_estandar, num_simulaciones)

# Promedio de rendimientos y montos acumulados
monto_promedio_acumulado = np.mean(resultados[:,-1])

# Escenarios pesimista y optimista
escenario_pesimista = np.min(resultados[:,-1])
escenario_optimista = np.max(resultados[:,-1])

# App en Streamlit
def app():
    st.title("Simulación de Montecarlo para Ahorros Universitarios")
    st.write("### Parámetros de la simulación")
    
    st.write(f"Ahorro anual: ${ahorro_anual}")
    st.write(f"Años de inversión: {años}")
    st.write(f"Rendimiento promedio anual: {tasa_promedio * 100}%")
    st.write(f"Desviación estándar: {desviacion_estandar * 100}%")
    
    # Mostrar los resultados
    st.write(f"Monto promedio acumulado: ${monto_promedio_acumulado:.2f}")
    st.write(f"Escenario pesimista: ${escenario_pesimista:.2f}")
    st.write(f"Escenario optimista: ${escenario_optimista:.2f}")
    
    # Gráficas
    st.write("### Gráficas")
    
    st.write("#### Rendimientos promedio por año")
    st.line_chart(resultados.mean(axis=0))
    
    st.write("#### Monto acumulado por año")
    for resultado in resultados:
        st.line_chart(resultado)

if __name__ == "__main__":
    app()
