import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.controllers.manager import Manager
from src.controllers.strategies.geometric_k import GeometricSIAK

def _get_tpm(n_bits):
    # Intenta cargar desde data/samples/N{n_bits}A.csv
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    METHOD2_ROOT = PROJECT_ROOT / "src" / "Method2_Dynamic_Programming_Reformulation"
    sys.path.insert(0, str(METHOD2_ROOT))
    path = METHOD2_ROOT / f"TPM_n{n_bits}.csv"
    if path.exists():
        return np.genfromtxt(path, delimiter=",")
    # Si no existe, crear TPM identidad (cada estado va a sí mismo)
    size = 2**n_bits
    tpm = np.eye(size, dtype=float)
    tpm = tpm / tpm.sum(axis=1, keepdims=True)
    return tpm

def _gestor(n_bits):
    estado = "1" + "0" * (n_bits - 1)
    return Manager(estado_inicial=estado)

def mostrar_perdidas(n, ks=[3,4,5]):
    tpm = _get_tpm(n)
    gest = _gestor(n)
    cond = alc = mec = "1" * n
    print(f"\n=== Sistema n={n} (k=2 ya conocido: 0.472656) ===")
    for k in ks:
        if k > n:
            print(f"k={k} no es posible (n={n})")
            continue
        kgeo = GeometricSIAK(gest)
        sol = kgeo.aplicar_estrategia(cond, alc, mec, tpm, k=k)
        print(f"k={k}: pérdida = {sol.perdida:.6f}")

if __name__ == "__main__":
    mostrar_perdidas(10)   # n=10
    # mostrar_perdidas(15) # si tienes TPM para n=15