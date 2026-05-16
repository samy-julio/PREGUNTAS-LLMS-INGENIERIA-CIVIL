import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def predecir_desercion(
    promedio,
    inasistencias,
    nivel_socioeconomico,
    horas_estudio
):
    """
    Predice deserción universitaria.
    """

    # Construir DataFrame
    X = pd.DataFrame({
        "promedio": promedio,
        "inasistencias": inasistencias,
        "nivel_socioeconomico": nivel_socioeconomico,
        "horas_estudio": horas_estudio
    })

    # Crear variable objetivo siguiendo la lógica del generador
    y = (
        (X["promedio"] < 2.8) |
        (X["inasistencias"] > 15) |
        (X["horas_estudio"] < 3)
    ).astype(int)

    # Imputación
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # Modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    # Predicción
    y_pred = model.predict(X_scaled)

    # Accuracy
    accuracy = accuracy_score(y, y_pred)

    # Variables importantes
    importancia = pd.Series(
        np.abs(model.coef_[0]),
        index=X.columns
    ).sort_values(ascending=False)

    return {
        "accuracy": accuracy,
        "variables_importantes": importancia
    }
