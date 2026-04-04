import pandas as pd
import numpy as np
import random

def generar_caso_de_uso_calcular_perfil_suelo():
    """
    Genera un caso de prueba para calcular_perfil_suelo
    """

    n = random.randint(10, 20)

    df = pd.DataFrame({
        'sample_id': np.random.choice(['S1', 'S2', 'S3'], size=n),
        'depth': np.random.uniform(1, 50, size=n),
        'unit_weight': np.random.uniform(15, 22, size=n),
        'moisture_content': np.random.uniform(5, 30, size=n)
    })

    input_data = {
        'df': df.copy()
    }

    # OUTPUT esperado
    grouped = df.groupby('sample_id').agg({
        'depth': 'max',
        'unit_weight': 'mean',
        'moisture_content': 'mean'
    }).reset_index()

    grouped = grouped.rename(columns={
        'depth': 'max_depth',
        'unit_weight': 'avg_unit_weight',
        'moisture_content': 'avg_moisture'
    })

    output_data = grouped

    return input_data, output_data
