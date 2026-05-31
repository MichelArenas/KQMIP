import sys
import time
import numpy as np
from pathlib import Path

# Obtener la ruta raíz del proyecto (dos niveles arriba)
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # sube 2 niveles desde Method2/... hasta GeoMIP/
SAMPLES_DIR = PROJECT_ROOT / "data" / "samples"

sys.path.insert(0, str(PROJECT_ROOT / "src" / "Method2_Dynamic_Programming_Reformulation" / "src"))
from src.controllers.manager import Manager
from src.controllers.strategies.geometric_k import GeometricSIAK

def test_performance():
    # Parámetros para un sistema de 10 nodos
    estado_inicial = "10000"   # 10 bits
    condicion = "1" * 5
    alcance = "1" * 5
    mecanismo = "1" * 5

    # Cargar TPM
    csv_path = SAMPLES_DIR / "N5A.csv"
    if not csv_path.exists():
        print(f"No se encuentra {csv_path}. Verifica la ruta.")
        return
    tpm = np.genfromtxt(csv_path, delimiter=",")
    print(f"TPM cargada: {csv_path} ({tpm.shape[1]} variables)")

    gestor = Manager(estado_inicial=estado_inicial)
    estrategia = GeometricSIAK(gestor)

    for k in (3, 4, 5):
        print(f"\n--- Probando k={k} ---")
        start = time.perf_counter()
        try:
            sol = estrategia.aplicar_estrategia(condicion, alcance, mecanismo, tpm, k=k)
            elapsed = time.perf_counter() - start
            print(f"Partición: {sol.particion[:100]}...")  # muestra inicio
            print(f"Pérdida: {sol.perdida}")
            print(f"Tiempo: {elapsed:.2f} segundos")
        except Exception as e:
            print(f"Error: {e}")
            elapsed = time.perf_counter() - start
            print(f"Tiempo hasta error: {elapsed:.2f} s")

if __name__ == "__main__":
    test_performance()