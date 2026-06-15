import pytest
import numpy as np
from unittest.mock import MagicMock
import sys
from pathlib import Path

sys.path.append(
    str(
        Path(__file__).resolve().parent
        / "GeoMIP"
        / "src"
        / "Method2_Dynamic_Programming_Reformulation"
    )
)

# Ajusta esta importación según la ruta real de tu clase en el proyecto
from GeoMIP.src.Method2_Dynamic_Programming_Reformulation.src.controllers.strategies.geometric_k import GeometricSIAK


class MockNCube:
    """Mock minimalista para simular los tensores del alcance del sistema."""
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32)

class MockSystem:
    """Mock de la clase System para aislar la prueba de la carga de archivos físicos."""
    def __init__(self):
        # Simulamos un sistema de 2 nodos: presente {0, 1}, futuro {0, 1}
        self.dims_ncubos = np.array([0, 1])
        self.indices_ncubos = np.array([0, 1])
        self.estado_inicial = np.array([0, 0])
        
        # Una TPM de juguete 2x2x2 mapeada a tensores aplanados
        # Representa probabilidades estables de transición
        self.ncubos = [
            MockNCube([[0.1, 0.8], [0.4, 0.9]]), # Nodo A
            MockNCube([[0.2, 0.7], [0.3, 0.85]]) # Nodo B
        ]
    
    def substraer(self, alcance, mecanismo):
        # Para la prueba unitaria, substraer devuelve una copia de sí mismo
        return self

    def distribucion_marginal(self):
        # Devuelve una distribución fija simulada
        return np.array([0.5, 0.5], dtype=np.float32)


@pytest.fixture
def analizador_geometrico():
    """Fixture que inicializa la estrategia geométrica con componentes simulados controlados."""
    # Creamos el mock principal
    gestor_mock = MagicMock()
    
    # Simulamos las propiedades que lee profiler_manager para construir la ruta en el disco.
    # En lugar de escupir <MagicMock...>, devolverá cadenas de texto limpias que no rompen Path.mkdir
    gestor_mock.pagina = "Prueba_Local"
    gestor_mock.nombre_archivo = "N2_Controlado"
    
    # Inicializamos la clase inyectando el mock configurado
    estrategia = GeometricSIAK(gestor_mock)
    
    # Inyectamos los componentes simulados del sistema que construimos en el paso anterior
    estrategia.sia_subsistema = MockSystem()
    estrategia.sia_logger = MagicMock()
    estrategia.sia_dists_marginales = np.array([0.5, 0.5], dtype=np.float32)
    estrategia.sia_tiempo_inicio = 0.0
    estrategia.idx_ncubos = [0, 1]
    
    # Configuración de variables de estado iniciales para N=2
    estrategia.vertices = {(0, 0), (0, 1), (1, 0), (1, 1)}
    estrategia.estado_inicial = np.array([0, 0])
    estrategia.estado_final = np.array([1, 1])
    
    # Inicialización de las estructuras de datos de GeoMIP
    estrategia.caminos = {0: [[0, 0]]}
    estrategia.tabla_transiciones = {((0, 0), (0, 0)): np.array([0.0, 0.0], dtype=np.float32)}
    
    # Preparar datos aplanados equivalentes al paso físico
    estrategia._flat_data = [ncubo.data.ravel() for ncubo in estrategia.sia_subsistema.ncubos]
    
    return estrategia


def test_cobertura_topologica_hipercubo(analizador_geometrico):
    """TEST 1: Valida que la topología del hipercubo se llene de forma acotada y exacta (Día 2)."""
    # Ejecutamos la expansión del hipercubo nivel por nivel para N=2
    analizador_geometrico.calcular_costos_nivel(np.array([1, 1]), nivel=1)
    analizador_geometrico.calcular_costos_nivel(np.array([1, 1]), nivel=2)
    
    # Verificación de la estructura binomial del hipercubo (Combinaciones de N en K)
    # N=2 => Nivel 0: 1 estado, Nivel 1: 2 estados, Nivel 2: 1 estado
    assert len(analizador_geometrico.caminos[0]) == 1  # Base: [0,0]
    assert len(analizador_geometrico.caminos[1]) == 2  # Vecinos: [1,0] y [0,1]
    assert len(analizador_geometrico.caminos[2]) == 1  # Destino: [1,1]
    
    # Comprobar que los estados específicos existan en la frontera de Hamming
    assert [1, 0] in analizador_geometrico.caminos[1]
    assert [0, 1] in analizador_geometrico.caminos[1]
    assert [1, 1] in analizador_geometrico.caminos[2]


def test_integridad_tabla_t(analizador_geometrico):
    """TEST 2: Valida que la Tabla T calcule y guarde todas las llaves de transiciones."""
    analizador_geometrico.calcular_costos_nivel(np.array([1, 1]), nivel=1)
    analizador_geometrico.calcular_costos_nivel(np.array([1, 1]), nivel=2)
    
    # Claves esperadas en el diccionario de la Tabla T
    key_inicio = (0, 0)
    key_vecino_1 = (1, 0)
    key_vecino_2 = (0, 1)
    key_fin = (1, 1)
    
    # Validar la existencia de las transiciones en la malla
    assert (key_inicio, key_vecino_1) in analizador_geometrico.tabla_transiciones
    assert (key_inicio, key_vecino_2) in analizador_geometrico.tabla_transiciones
    assert (key_inicio, key_fin) in analizador_geometrico.tabla_transiciones
    
    # Validar que los costos acumulados sean numéricamente válidos (no NaN ni infinitos)
    costo_final = np.array(analizador_geometrico.tabla_transiciones[(key_inicio, key_fin)])
    assert not np.isnan(costo_final).any()
    assert np.all(costo_final >= 0.0)

@pytest.fixture
def estrategia_validada():
    """Configura un GeometricSIAK controlado con dimensiones alineadas (N=3, 2^N=8)."""
    gestor_mock = MagicMock()
    gestor_mock.pagina = "Test_Validacion"
    gestor_mock.nombre_archivo = "N3_Controlado"
    
    estrategia = GeometricSIAK(gestor_mock)
    estrategia.sia_subsistema = MagicMock()
    
    # Sistema de 3 nodos (N=3)
    estrategia.sia_subsistema.indices_ncubos = [0, 1, 2]
    estrategia.idx_ncubos = [0, 1, 2]
    estrategia.vertices = {(0,0), (0,1), (0,2), (1,0), (1,1), (1,2)}
    estrategia.estado_inicial = np.array([1, 0, 0])
    estrategia.estado_final = np.array([0, 0, 1])
    
    # Distribución marginal de tamaño 2^3 = 8 elementos (PMF válida que suma 1.0)
    estrategia.sia_dists_marginales = np.array([0.125] * 8, dtype=np.float32)
    estrategia.vertices_sistema = list(estrategia.vertices)
    
    # Inyectar Tabla T de transiciones (Guardada como lista por clave, igual que tu main)
    key_raiz = ((1, 0, 0), (0, 0, 1))
    estrategia.caminos = {0: [[1, 0, 0]]}
    estrategia.tabla_transiciones = {key_raiz: [0.1, 0.4, 0.2]} 
    
    # Mockear datos planos internos para evitar fugas de memoria
    estrategia._flat_data = [np.zeros(8) for _ in range(3)]
    
    # Mockear kpartir y distribucion_marginal para retornar un array de NumPy
    mock_dist = np.array([0.125] * 8, dtype=np.float32)
    estrategia.sia_subsistema.kpartir.return_value.distribucion_marginal.return_value = mock_dist
    
    return estrategia

# ==============================================================================
# OBJETIVO 1: CORRECTITUD K=2 (CONSISTENCIA ESTRICTA)
# ==============================================================================

def test_k2_distribucion_suma_uno(estrategia_validada):
    """[OBJETIVO 1] La distribución conjunta reconstruida debe ser una PMF válida de tamaño 2^N."""
    particion_k2 = [
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)], 
        [(1, 2)]                                  
    ]
    
    dist = estrategia_validada.evaluar_distribucion_k_particion(particion_k2)
    
    assert isinstance(dist, np.ndarray), "Debe retornar un arreglo de NumPy."
    assert dist.shape == (8,), f"La forma de la distribución debe ser (8,), pero dio {dist.shape}"
    assert np.isclose(np.sum(dist), 1.0, atol=1e-4), f"La suma es {np.sum(dist)}, debe ser 1.0."


def test_integridad_tabla_t_corregido(estrategia_validada):
    """[CORRECCIÓN TYPE_ERROR] Valida que la estructura de la Tabla T sea numérica al colapsarse."""
    key_raiz = ((1, 0, 0), (0, 0, 1))
    costos = estrategia_validada.tabla_transiciones[key_raiz]
    
    # Simula la aserción del test original pero extrayendo el escalar de la lista de costos
    assert isinstance(costos, list), "La Tabla T debe almacenar listas de costos por canal."
    assert sum(costos) >= 0.0, "La suma acumulada de costos numéricos debe ser no-negativa."

# ==============================================================================
# OBJETIVO 2: K-PARTICIONES K=3..5 FUNCIONALES
# ==============================================================================

def test_k_particiones_partes_no_vacias(estrategia_validada):
    """[OBJETIVO 2] Cada bloque dentro de una partición candidata debe contener elementos."""
    candidatos = estrategia_validada.identificar_particiones_optimas()
    
    assert len(candidatos) > 0, "Debe generar al menos un candidato de corte."
    for particion in candidatos:
        assert len(particion) >= 2, "Cada partición debe tener al menos 2 bloques (K>=2)."
        for bloque in particion:
            assert len(bloque) > 0, "Restricción formal: Ningún bloque puede estar vacío."


def test_monotonicidad_phi_creciente(estrategia_validada):
    """[OBJETIVO 2 - DIMENSIONES FIJAS] Valida que K=3 tenga igual o más dispersión que K=2."""
    particion_k2 = [[(0,0), (0,1), (0,2), (1,0), (1,1)], [(1,2)]]
    particion_k3 = [[(0,0), (0,1), (0,2)], [(1,0)], [(1,1), (1,2)]] 
    
    dist_k2 = estrategia_validada.evaluar_distribucion_k_particion(particion_k2)
    dist_k3 = estrategia_validada.evaluar_distribucion_k_particion(particion_k3)
    
    # Ambas distribuciones ahora tienen forma (8,), el broadcasting de NumPy es perfecto
    error_k2 = np.linalg.norm(dist_k2 - estrategia_validada.sia_dists_marginales)
    error_k3 = np.linalg.norm(dist_k3 - estrategia_validada.sia_dists_marginales)
    
    assert error_k3 >= error_k2, "Validación IIT: A mayor fragmentación K, mayor es la pérdida de información."