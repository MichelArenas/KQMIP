import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.controllers.manager import Manager
from src.controllers.strategies.geometric_k import GeometricSIAK
from src.funcs.base import emd_efecto

# ------------------------------------------------------------
# Utilidades
# ------------------------------------------------------------
def generar_todas_k_particiones(elementos, k):
    """Retorna lista de particiones, cada partición es lista de listas."""
    if k == 1:
        return [[elementos]]
    if len(elementos) == k:
        return [[[x] for x in elementos]]
    if len(elementos) < k or k < 1:
        return []
    primer = elementos[0]
    resto = elementos[1:]
    res = []
    for p in generar_todas_k_particiones(resto, k-1):
        res.append([[primer]] + p)
    for p in generar_todas_k_particiones(resto, k):
        for i in range(len(p)):
            nueva = [list(grupo) for grupo in p]
            nueva[i].append(primer)
            res.append(nueva)
    return res

def evaluar_particion_exhaustiva(subsistema, particion):
    """Evalúa una partición (lista de grupos de índices) usando kpartir."""
    grupos = []
    for grupo in particion:
        fut = np.array(grupo, dtype=np.int8)
        pres = np.array(grupo, dtype=np.int8)
        grupos.append((fut, pres))
    dist = subsistema.kpartir(grupos).distribucion_marginal()
    return emd_efecto(dist, subsistema.distribucion_marginal())

def normalizar_particion(particion):
    """Devuelve representación canónica (tupla de tuplas ordenadas)."""
    return tuple(sorted(tuple(sorted(g)) for g in particion))

# ------------------------------------------------------------
# Validación para un sistema dado
# ------------------------------------------------------------
def validar_para_sistema(tpm_path, estado_inicial, n_vars, ks=[3,4,5]):
    tpm = np.genfromtxt(tpm_path, delimiter=",")
    gestor = Manager(estado_inicial)
    # Convención: '1' conserva, '0' marginaliza
    condicion = "1" * n_vars
    alcance = "1" * n_vars
    mecanismo = "1" * n_vars

    # Instanciamos una sola estrategia para evitar recalcular el subsistema varias veces
    estrategia = GeometricSIAK(gestor)
    estrategia.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)
    subsistema = estrategia.sia_subsistema
    indices = list(range(n_vars))

    resultados = {}
    for k in ks:
        if k > n_vars:
            print(f"  k={k} no es posible (n={n_vars}), saltando.")
            continue
        print(f"\n--- Validando k={k} para n={n_vars} ---")
        # 1. Exhaustivo
        todas = generar_todas_k_particiones(indices, k)
        print(f"  Número de particiones exhaustivas: {len(todas)}")
        mejor_phi = float('inf')
        mejor_particion = None
        for part in todas:
            phi = evaluar_particion_exhaustiva(subsistema, part)
            if phi < mejor_phi:
                mejor_phi = phi
                mejor_particion = part
        # 2. Algoritmo KGeoMIP
        sol = estrategia.aplicar_estrategia(condicion, alcance, mecanismo, tpm, k=k)
        phi_alg = sol.perdida
        # Obtener la partición real encontrada (desde _candidatos_evaluados)
        mejor_clave = min(estrategia.memoria_particiones, key=lambda c: estrategia.memoria_particiones[c][0])
        particion_alg_raw = estrategia._candidatos_evaluados[mejor_clave]
        grupos_alg = [list(fut) for _, fut in particion_alg_raw]
        norm_alg = normalizar_particion(grupos_alg)
        norm_opt = normalizar_particion(mejor_particion)
        acierto = (norm_alg == norm_opt)
        error_rel = abs(phi_alg - mejor_phi) / mejor_phi if mejor_phi > 0 else (0 if phi_alg == 0 else 1)
        resultados[k] = {
            "phi_opt": mejor_phi,
            "phi_alg": phi_alg,
            "acierto": acierto,
            "error_rel": error_rel,
            "num_particiones": len(todas)
        }
        print(f"  φ óptimo: {mejor_phi:.6f}, φ algoritmo: {phi_alg:.6f}, acierto: {acierto}, error_rel: {error_rel:.4f}")
    return resultados

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    # Ajusta la ruta según tu estructura real
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    data_dir = PROJECT_ROOT / "data" / "samples"
    # Si no encuentras los CSV, prueba con:
    # data_dir = Path(__file__).parent.parent / "data" / "samples"
    sistemas = [
        (data_dir / "N3A.csv", "100", 3),
        (data_dir / "N4A.csv", "1000", 4),
        (data_dir / "N5A.csv", "10000", 5),
    ]
    for csv_path, estado, n in sistemas:
        if not csv_path.exists():
            print(f"Advertencia: {csv_path} no existe. Saltando n={n}")
            continue
        print(f"\n========== Sistema n={n} ==========")
        resultados = validar_para_sistema(csv_path, estado, n, ks=[3,4,5])
        for k, res in resultados.items():
            print(f"k={k}: acierto={res['acierto']}, error_rel={res['error_rel']:.6f}, φ_opt={res['phi_opt']:.6f}, φ_alg={res['phi_alg']:.6f}")