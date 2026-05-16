import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2

def calcular_scores_chi2(df, columnas, columna_objetivo):
    """
    Calcula scores chi-cuadrado para variables predictoras
    respecto a una variable objetivo categórica.
    """

    # Seleccionar variables predictoras y objetivo
    X = df[columnas].to_numpy()
    y = df[columna_objetivo].to_numpy()

    # Escalar al rango [0,1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Calcular scores chi-cuadrado
    scores, _ = chi2(X_scaled, y)

    return scores
