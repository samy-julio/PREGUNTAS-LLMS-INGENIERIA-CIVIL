import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def predecir_desercion(df, deserta):
    """
    Entrena un modelo para predecir deserción universitaria.
    
    Parámetros:
    - df: DataFrame con variables predictoras
    - deserta: array o serie con variable objetivo
    
    Retorna:
    - accuracy
    - variables_importantes
    """

    # Convertir a DataFrame por seguridad
    X = pd.DataFrame(df).copy()
    y = np.array(deserta)

    # -------------------------------------------------
    # 1. Limpiar datos faltantes
    # -------------------------------------------------
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # -------------------------------------------------
    # 2. Estandarizar variables
    # -------------------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # -------------------------------------------------
    # 3. Entrenar modelo
    # -------------------------------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_scaled, y)

    # -------------------------------------------------
    # 4. Predicciones
    # -------------------------------------------------
    y_pred = model.predict(X_scaled)

    # -------------------------------------------------
    # 5. Métricas
    # -------------------------------------------------
    accuracy = accuracy_score(y, y_pred)

    # -------------------------------------------------
    # 6. Variables más importantes
    # -------------------------------------------------
    importancia = pd.Series(
        np.abs(model.coef_[0]),
        index=X.columns
    ).sort_values(ascending=False)

    return {
        "accuracy": accuracy,
        "variables_importantes": importancia
    }
