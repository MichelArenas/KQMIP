import numpy as np
from src.controllers.manager import Manager
from src.strategies.q_nodes_k import ParticionadorQ

def probar_k_para_red(estado_inicial, condicion, alcance, mecanismo, k):
    print(f"\n=== Probando k={k} ===")
    gestor = Manager(estado_inicial)
    tpm = gestor.cargar_red()
    estrategia = ParticionadorQ(tpm)
    sol = estrategia.aplicar_estrategia(
        estado_inicial, condicion, alcance, mecanismo, k=k
    )
    print(f"Partición: {sol.particion}")
    print(f"Pérdida (EMD): {sol.perdida}")
    #print(f"Tiempo: {sol.tiempo_total:.4f} s")
    return sol

if __name__ == "__main__":
    # Parámetros (ajusta según tu red)
    estado_inicial = "100"
    condicion = "111"
    alcance = "111"
    mecanismo = "111"
    
    # Probar k=2 (debe dar similar a QNodes original)
    probar_k_para_red(estado_inicial, condicion, alcance, mecanismo, k=2)
    
    # Probar k=3,4 (si el número de variables lo permite)
    # Si la red tiene menos de 3 variables, k no puede ser mayor.
    # Puedes verificar con gestor.cargar_red().shape[1] (número de nodos)
    gestor_temp = Manager(estado_inicial)
    tpm_temp = gestor_temp.cargar_red()
    n_vars = tpm_temp.shape[1]  # número de variables futuras
    print(f"\nNúmero de variables del sistema: {n_vars}")
    if n_vars >= 3:
        probar_k_para_red(estado_inicial, condicion, alcance, mecanismo, k=3)
    if n_vars >= 4:
        probar_k_para_red(estado_inicial, condicion, alcance, mecanismo, k=4)
    if n_vars >= 5:
        probar_k_para_red(estado_inicial, condicion, alcance, mecanismo, k=5)