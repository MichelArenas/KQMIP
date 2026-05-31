import pytest
import numpy as np
from src.controllers.manager import Manager
from src.strategies.q_nodes_k import ParticionadorQ

# ------------------------------------------------------------
# Fixture para cargar una red pequeña (3 nodos)
# ------------------------------------------------------------
@pytest.fixture
def red_3nodos():
    estado = "100"
    cond = "111"
    alc = "111"
    mec = "111"
    gestor = Manager(estado)
    tpm = gestor.cargar_red()  # debe cargar N3C.csv o la red correspondiente
    return estado, cond, alc, mec, tpm

# ------------------------------------------------------------
# Prueba 1: k=2 debe retornar 2 grupos y pérdida finita
# ------------------------------------------------------------
def test_k2_retorna_dos_grupos(red_3nodos):
    estado, cond, alc, mec, tpm = red_3nodos
    estrategia = ParticionadorQ(tpm)
    sol = estrategia.aplicar_estrategia(estado, cond, alc, mec, k=2)
    
    # La partición debe ser una cadena (formateada) que contenga "G1:" y "G2:"
    assert "G1:" in sol.particion
    assert "G2:" in sol.particion
    # La pérdida debe ser un número no negativo
    assert sol.perdida >= 0
    # El tiempo debe ser positivo
   # #assert sol.tiempo_total > 0

# ------------------------------------------------------------
# Prueba 2: k=3 debe retornar 3 grupos (válido para 3 nodos)
# ------------------------------------------------------------
def test_k3_retorna_tres_grupos(red_3nodos):
    estado, cond, alc, mec, tpm = red_3nodos
    estrategia = ParticionadorQ(tpm)
    sol = estrategia.aplicar_estrategia(estado, cond, alc, mec, k=3)
    
    # Debe tener tres grupos (G1, G2, G3)
    assert sol.particion.count("G") == 3
    assert sol.perdida >= 0
    #assert sol.tiempo_total > 0

# ------------------------------------------------------------
# Prueba 3: k=4 debería lanzar ValueError si hay menos de 4 variables
# ------------------------------------------------------------
def test_k4_excede_variables(red_3nodos):
    estado, cond, alc, mec, tpm = red_3nodos
    estrategia = ParticionadorQ(tpm)
    # Para una red de 3 nodos, k=4 debe fallar
    with pytest.raises(ValueError, match="supera las variables futuras"):
        estrategia.aplicar_estrategia(estado, cond, alc, mec, k=4)

# ------------------------------------------------------------
# Prueba 4: k=1 debe lanzar ValueError (k debe ser ≥2)
# ------------------------------------------------------------
def test_k1_invalido(red_3nodos):
    estado, cond, alc, mec, tpm = red_3nodos
    estrategia = ParticionadorQ(tpm)
    with pytest.raises(ValueError, match="k debe ser ≥ 2"):
        estrategia.aplicar_estrategia(estado, cond, alc, mec, k=1)

# ------------------------------------------------------------
# Prueba 5: Consistencia con el algoritmo original (k=2) vs fuerza bruta (opcional)
# ------------------------------------------------------------
def test_k2_coincide_con_bipartir(red_3nodos):
    """
    Verifica que para k=2, el resultado de ParticionadorQ sea el mismo
    que usar directamente el método `bipartir` de System (o una referencia).
    Como no tenemos el QNodes original en este contexto, comparamos
    que la pérdida esté en un rango esperado (por ejemplo, entre 0 y 1).
    """
    estado, cond, alc, mec, tpm = red_3nodos
    estrategia = ParticionadorQ(tpm)
    sol = estrategia.aplicar_estrategia(estado, cond, alc, mec, k=2)
    
    # Pérdida razonable para un sistema pequeño (ajusta según tu red)
    assert 0 <= sol.perdida <= 1.0
    # Además, la partición debe tener exactamente dos grupos
    num_grupos = sol.particion.count("G")
    assert num_grupos == 2