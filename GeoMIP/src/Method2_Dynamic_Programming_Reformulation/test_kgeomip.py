import pytest
import numpy as np
from pathlib import Path
from src.controllers.manager import Manager
from src.controllers.strategies.geometric_k import GeometricSIAK

# ------------------------------------------------------------
# Configuración de rutas
# ------------------------------------------------------------
# Ruta a la carpeta samples (ajústala según tu proyecto)
CSV_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"
print(CSV_DIR)
print(CSV_DIR.exists())
# Si el archivo de pruebas está en otra ubicación, cambia la ruta base.
# Por ejemplo, si está en la raíz: CSV_DIR = Path("geomip/data/samples")

def cargar_tpm(nombre_csv):
    """Carga un CSV desde la carpeta samples."""
    path = CSV_DIR / nombre_csv
    if not path.exists():
        raise FileNotFoundError(f"No se encuentra {path}")
    return np.genfromtxt(path, delimiter=",")

# ------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------
@pytest.fixture
def sistema_3nodos():
    """Carga TPM de 3 nodos (N3A.csv) y devuelve gestor y tpm."""
    estado = "100"
    gestor = Manager(estado)
    tpm = cargar_tpm("N3A.csv")
    return gestor, tpm

@pytest.fixture
def sistema_4nodos():
    estado = "1000"
    gestor = Manager(estado)
    tpm = cargar_tpm("N4A.csv")  # Asegúrate de que exista N4A.csv
    return gestor, tpm

# ------------------------------------------------------------
# Utilidades exhaustivas
# ------------------------------------------------------------
def generar_todas_k_particiones(elementos, k):
    """Genera todas las k‑particiones de una lista de elementos."""
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
    grupos_kpartir = []
    for grupo in particion:
        fut = np.array(grupo, dtype=np.int8)
        pres = np.array(grupo, dtype=np.int8)
        grupos_kpartir.append((fut, pres))
    dist = subsistema.kpartir(grupos_kpartir).distribucion_marginal()
    dist_original = subsistema.distribucion_marginal()
    from src.funcs.base import emd_efecto
    return emd_efecto(dist, dist_original)

# ------------------------------------------------------------
# Pruebas
# ------------------------------------------------------------
def test_k3_en_sistema_3nodos(sistema_3nodos):
    gestor, tpm = sistema_3nodos
    condicion = "111"
    alcance = "111"
    mecanismo = "111"
    k = 3

    estrategia = GeometricSIAK(gestor)
    sol = estrategia.aplicar_estrategia(condicion, alcance, mecanismo, tpm, k=k)

    indices = [0, 1, 2]
    todas = generar_todas_k_particiones(indices, k)
    assert len(todas) == 1
    perdida_exhaustiva = evaluar_particion_exhaustiva(estrategia.sia_subsistema, todas[0])
    assert abs(sol.perdida - perdida_exhaustiva) < 1e-6
    assert sol.particion.startswith("k=3 | G1:")

def test_k4_en_sistema_4nodos(sistema_4nodos):
    gestor, tpm = sistema_4nodos
    condicion = "1111"
    alcance = "1111"
    mecanismo = "1111"
    k = 4

    estrategia = GeometricSIAK(gestor)
    sol = estrategia.aplicar_estrategia(condicion, alcance, mecanismo, tpm, k=k)

    indices = [0, 1, 2, 3]
    todas = generar_todas_k_particiones(indices, k)
    assert len(todas) == 1
    perdida_exhaustiva = evaluar_particion_exhaustiva(estrategia.sia_subsistema, todas[0])
    assert abs(sol.perdida - perdida_exhaustiva) < 1e-6

def test_generacion_candidatos_k3(sistema_3nodos):
    gestor, tpm = sistema_3nodos
    estrategia = GeometricSIAK(gestor)
    # Preparamos el subsistema manualmente (sin ejecutar find_mip completo)
    condicion = "111"
    alcance = "111"
    mecanismo = "111"
    estrategia.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)
    dims = estrategia.sia_subsistema.dims_ncubos
    estrategia.estado_inicial = estrategia.sia_subsistema.estado_inicial[dims]
    estrategia.estado_final = 1 - estrategia.estado_inicial
    estrategia.idx_ncubos = list(range(len(estrategia.sia_subsistema.indices_ncubos)))
    estrategia.caminos = {0: [estrategia.estado_inicial.tolist()]}
    # Necesitamos la tabla de costos para identificar_particiones_optimas.
    # La forma más fácil es llamar a calcular_costos_nivel, pero requiere _flat_data.
    estrategia._flat_data = [ncubo.data.ravel() for ncubo in estrategia.sia_subsistema.ncubos]
    # Construir la tabla de costos completa para el subsistema (costoso pero una sola vez)
    for nivel in range(1, len(estrategia.estado_inicial) + 1):
        estrategia.calcular_costos_nivel(estrategia.estado_final, nivel)
    estrategia.k = 3
    candidatos = estrategia.identificar_particiones_optimas()
    assert len(candidatos) > 0
    for cand in candidatos:
        assert len(cand) == 3

def test_kpartir_evaluacion_valida(sistema_3nodos):
    gestor, tpm = sistema_3nodos
    estrategia = GeometricSIAK(gestor)
    condicion = "111"
    alcance = "111"
    mecanismo = "111"
    sol = estrategia.aplicar_estrategia(condicion, alcance, mecanismo, tpm, k=3)
    assert isinstance(sol.distribucion_particion, np.ndarray)
    assert sol.distribucion_particion.size > 0
    assert np.isfinite(sol.perdida)

def test_k_mayor_que_variables_futuras(sistema_3nodos):
    gestor, tpm = sistema_3nodos
    estrategia = GeometricSIAK(gestor)
    condicion = "111"
    alcance = "111"
    mecanismo = "111"
    with pytest.raises(ValueError, match="supera las variables futuras"):
        estrategia.aplicar_estrategia(condicion, alcance, mecanismo, tpm, k=4)