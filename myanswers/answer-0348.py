import numpy as np
from sklearn.decomposition import IncrementalPCA

def reducir_dimensionalidad_incremental(
    datos,
    n_componentes,
    batch_size
):
    """
    Reduce dimensionalidad usando IncrementalPCA
    procesando los datos por batches.
    """

    # Inicializar modelo
    modelo = IncrementalPCA(
        n_components=n_componentes,
        batch_size=batch_size
    )

    n_filas = len(datos)

    # Ajuste incremental usando partial_fit
    for i in range(0, n_filas, batch_size):
        batch = datos.iloc[i:i + batch_size]
        modelo.partial_fit(batch)

    # Transformar todos los datos
    X_transformado = modelo.transform(datos)

    return X_transformado
