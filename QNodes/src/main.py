from src.controllers.manager import Manager

# 👇 Importación de estrategias 👇 #
from src.strategies.force import BruteForce


def iniciar():
    """Punto de entrada"""

    # ABCD #
    estado_inicial = "100"
    condiciones =    "111"
    alcance =        "111"
    mecanismo =      "111"

    gestor_redes = Manager(estado_inicial)
    mpt = gestor_redes.cargar_red()

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_bf = BruteForce(mpt)

    sia_cero = analizador_bf.aplicar_estrategia(
        estado_inicial,
        condiciones,
        alcance,
        mecanismo,
    )
    print(sia_cero)
