import numpy as np
from sklearn.impute import SimpleImputer

def preparar_matriz_con_imputacion(df, target_col):
    """
    Prepara una matriz imputando valores faltantes.
    """

    # Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col].to_numpy()

    # Imputar valores faltantes
    imputer = SimpleImputer(strategy="mean")
    X_imputed = imputer.fit_transform(X)

    # Retornar resultado
    return (X_imputed, y)
