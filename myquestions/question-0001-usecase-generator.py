import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_calcular_alertas_creciente():
    """
    Genera un caso de prueba para calcular_alertas_creciente
    """

    # 1. Datos aleatorios
    n = random.randint(8, 15)
    station_ids = np.random.choice(['A', 'B'], size=n)

    timestamps = pd.date_range("2020-01-01", periods=n, freq="H")
    water_levels = np.random.uniform(0, 10, size=n)

    df = pd.DataFrame({
        'station_id': station_ids,
        'timestamp': timestamps,
        'water_level': water_levels
    })

    umbral = random.uniform(0.5, 2.0)

    input_data = {
        'df': df.copy(),
        'umbral': umbral
    }

    # 2. OUTPUT esperado
    df_sorted = df.copy()
    df_sorted['timestamp'] = pd.to_datetime(df_sorted['timestamp'])
    df_sorted = df_sorted.sort_values(['station_id', 'timestamp'])

    df_sorted['incremento'] = df_sorted.groupby('station_id')['water_level'].diff()
    df_sorted['alerta'] = df_sorted['incremento'] > umbral

    output_data = df_sorted

    return input_data, output_data
