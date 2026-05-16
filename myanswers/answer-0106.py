import numpy as np

def predecir_desercion(
    promedio,
    inasistencias,
    nivel_socioeconomico,
    horas_estudio
):
    """
    Predice deserción universitaria.
    """

    deserta = (
        (np.array(promedio) < 2.8) |
        (np.array(inasistencias) > 15) |
        (np.array(horas_estudio) < 3)
    ).astype(int)

    return deserta
