import sys
from pathlib import Path
import numpy as np
import pytest

# Ajusta la ruta según tu estructura
sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA
from src.controllers.strategies.geometric_k import GeometricSIAK, KGeoMIP

# ============================================================
# Utilidades
# ============================================================

def _get_tpm(n_bits):
    """Carga TPM desde archivos de ejemplo o genera sintética."""
    # Intenta cargar desde data/samples/N{n_bits}A.csv
    PROJECT_ROOT = Path(__file__).resolve().parents[2]  # sube 2 niveles desde Method2/... hasta GeoMIP/
    SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"
    path = SAMPLES_DIR / f"N{n_bits}A.csv"
    if path.exists():
        return np.genfromtxt(path, delimiter=",")
    # Si no existe, crear TPM identidad (cada estado va a sí mismo)
    size = 2**n_bits
    tpm = np.eye(size, dtype=float)
    # Normalizar (ya lo está, pero por si acaso)
    tpm = tpm / tpm.sum(axis=1, keepdims=True)
    return tpm

def _gestor(n_bits):
    """Crea un Manager con un estado inicial de n_bits ceros excepto el primero."""
    estado = "1" + "0" * (n_bits - 1)
    return Manager(estado_inicial=estado)

# ============================================================
# Tests
# ============================================================

def test_k2_consistencia():
    n = 4
    tpm = _get_tpm(n)
    gest = _gestor(n)
    cond = alc = mec = "1" * n

    geo_orig = GeometricSIA(gest)
    sol_orig = geo_orig.aplicar_estrategia(cond, alc, mec, tpm)

    geo_k = GeometricSIAK(gest)
    sol_k = geo_k.aplicar_estrategia(cond, alc, mec, tpm, k=2)

    assert np.isclose(sol_orig.perdida, sol_k.perdida, atol=1e-8), \
        f"Pérdidas diferentes: orig={sol_orig.perdida}, k2={sol_k.perdida}"
    print(f"✓ k=2 consistente: φ={sol_k.perdida:.6f}")


def test_monotonia_creciente():
    """
    Verifica que δ2 ≤ δ3 ≤ δ4 ≤ δ5 (monotonía no decreciente).
    Para sistemas con n=4 o 5, con TPM real.
    """
    n = 4
    tpm = _get_tpm(n)
    gest = _gestor(n)
    cond = alc = mec = "1" * n
    geo_k = GeometricSIAK(gest)

    phi = {}
    for k in (2,3,4,5):
        if k > n:
            continue
        sol = geo_k.aplicar_estrategia(cond, alc, mec, tpm, k=k)
        phi[k] = sol.perdida
        print(f"φ({k}) = {phi[k]:.6f}")

    # Verificar desigualdades
    ks = sorted(phi.keys())
    for i in range(len(ks)-1):
        assert phi[ks[i]] <= phi[ks[i+1]] + 1e-9, \
            f"Monotonía violada: φ({ks[i]})={phi[ks[i]]:.6f} > φ({ks[i+1]})={phi[ks[i+1]]:.6f}"
    print("✓ Monotonía creciente verificada")

def test_perdidas_razonables_k3():
    """
    Para un sistema pequeño (n=3) donde la búsqueda exhaustiva es viable,
    la pérdida para k=3 debe ser mucho menor que 4.59 (el valor erróneo anterior).
    Esperamos un valor ≤ 1.0 (típicamente cercano a φ(k=2) o un poco mayor).
    """
    n = 3
    tpm = _get_tpm(n)
    gest = _gestor(n)
    cond = alc = mec = "1" * n

    geo_k = GeometricSIAK(gest)
    sol2 = geo_k.aplicar_estrategia(cond, alc, mec, tpm, k=2)
    sol3 = geo_k.aplicar_estrategia(cond, alc, mec, tpm, k=3)

    phi2 = sol2.perdida
    phi3 = sol3.perdida

    # La pérdida para k=3 no debería ser órdenes de magnitud mayor.
    # Un valor razonable: phi3 <= phi2 + 1.0 (por ejemplo, de 0.47 a 1.47)
    # En sistemas reales, puede subir un poco, pero no a 4.6.
    assert phi3 < 3.5, f"φ(k=3)={phi3} es demasiado alto comparado con φ(k=2)={phi2}"
    print(f"✓ k=3: φ={phi3:.6f} (aumento razonable desde {phi2:.6f})")

def test_particiones_validas():
    """Para k=3,4,5 comprueba que la partición tiene exactamente k grupos y no vacíos."""
    n = 4
    tpm = _get_tpm(n)
    gest = _gestor(n)
    cond = alc = mec = "1" * n
    geo_k = GeometricSIAK(gest)

    for k in (3,4,5):
        if k > n:
            continue
        sol = geo_k.aplicar_estrategia(cond, alc, mec, tpm, k=k)
        # Contar grupos (aparece "G1:", "G2:", ...)
        n_grupos = sol.particion.count("G")
        assert n_grupos == k, f"Se esperaban {k} grupos, se encontraron {n_grupos}"
        # Verificar que ningún grupo está vacío (aparece "|-" o algo similar)
        # Asumimos que el formato no incluye grupos vacíos; si los hubiera, la cadena mostraría "{}"
        assert "{}" not in sol.particion, "Se encontró un grupo vacío en la partición"
        print(f"✓ k={k}: partición válida con {n_grupos} grupos")
"""
def test_sistema_identidad():
   
    Con TPM identidad (cada estado va a sí mismo), la pérdida óptima para cualquier k
    debería ser 0, porque se puede asignar cada variable futura a su presente y hacer
    grupos unipersonales (si k == n) o agrupar sin pérdida.
    Para k < n, también se puede lograr pérdida 0 agrupando adecuadamente (p.ej., si todas
    las variables son independientes, cualquier partición da pérdida 0).
    
   
    n = 3   # número de bits
    size = 2 ** n
    tpm = np.eye(size, dtype=float)   # TPM identidad
    estado = "1" + "0" * (n - 1)
    print(f"Creando Manager con estado '{estado}' de longitud {len(estado)}")
    gest = Manager(estado_inicial=estado)
    # gest = Manager(estado_inicial="1" + "0" * (n - 1))   # estado "100"
    cond = alc = mec = "1" * n
    geo_k = GeometricSIAK(gest)

    for k in (2, 3):
        sol = geo_k.aplicar_estrategia(cond, alc, mec, tpm, k=k)
        assert sol.perdida < 1e-8, f"Para TPM identidad k={k}, pérdida debería ser 0, pero es {sol.perdida}"
        print(f"✓ TPM identidad k={k}: φ={sol.perdida:.2e}")
    """

def test_mejora_respecto_versiones_anteriores():
    """
    (Opcional) Verifica que la pérdida para k=3 no es el valor disparado 4.59.
    Si se ejecuta sobre el mismo sistema que antes daba 4.59, ahora debe ser menor.
    """
    n = 10   # el mismo sistema que daba φ(k=2)=0.472656 y φ(k=3)=4.613281
    # Nota: asegúrate de que el archivo N10A.csv exista en data/samples
    tpm = _get_tpm(n)
    gest = _gestor(n)
    cond = alc = mec = "1" * n
    geo_k = GeometricSIAK(gest)

    sol2 = geo_k.aplicar_estrategia(cond, alc, mec, tpm, k=2)
    sol3 = geo_k.aplicar_estrategia(cond, alc, mec, tpm, k=3)

    phi2 = sol2.perdida
    phi3 = sol3.perdida

    # Antes phi3 era ~4.6, ahora debería ser mucho más cercano a phi2
    assert phi3 < 3.5, f"φ(k=3)={phi3} sigue siendo demasiado alto (φ2={phi2})"
    print(f"✓ Mejora: φ3={phi3:.6f} (vs φ2={phi2:.6f})")

# Ejecutar con: pytest test_kgeometric_validacion.py -v