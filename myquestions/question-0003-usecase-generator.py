import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def generar_caso_de_uso_evaluar_modelo_pavimento():
    """
    Genera un caso de prueba para evaluar_modelo_pavimento
    """

    n = random.randint(20, 40)
    n_features = random.randint(3, 6)

    data = np.random.randn(n, n_features)
    cols = [f'feature_{i}' for i in range(n_features)]

    df = pd.DataFrame(data, columns=cols)
    target_col = 'deflection'
    df[target_col] = np.random.randn(n)

    input_data = {
        'df': df.copy(),
        'target_col': target_col
    }

    # OUTPUT esperado
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    output_data = mae

    return input_data, output_data
