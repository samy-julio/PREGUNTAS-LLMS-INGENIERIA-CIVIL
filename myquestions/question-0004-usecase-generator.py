import pandas as pd
import numpy as np
import random
from sklearn.cluster import KMeans

def generar_caso_de_uso_agrupar_danos():
    """
    Genera un caso de prueba para agrupar_danos
    """

    n = random.randint(10, 30)

    df = pd.DataFrame({
        'x': np.random.uniform(0, 100, size=n),
        'y': np.random.uniform(0, 100, size=n)
    })

    n_clusters = random.randint(2, 4)

    input_data = {
        'df': df.copy(),
        'n_clusters': n_clusters
    }

    # OUTPUT esperado
    X = df[['x', 'y']]

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    labels = kmeans.fit_predict(X)

    output_data = labels

    return input_data, output_data
