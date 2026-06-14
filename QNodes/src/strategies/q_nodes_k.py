"""
particionador_q.py — Partición Q para k-grupos (peeling greedy secuencial).
Adaptación del algoritmo QNodes original.

Universidad de Caldas — Proyecto K-QGMIP — 2026-1
"""

import time
from typing import Optional, List, Tuple
import numpy as np
from numpy.typing import NDArray

from src.funcs.iit import emd_efecto, ABECEDARY
from src.middlewares.slogger import SafeLogger
from src.middlewares.profile import gestor_perfilado, profile
from src.models.base.sia import SIA
from src.models.core.solution import Solution
from src.constants.models import QNODES_ANALYSIS_TAG
from src.constants.base import (
    COLS_IDX,
    INT_ZERO,
    TYPE_TAG,
    NET_LABEL,
    INFTY_POS,
    EFFECT,
    ACTUAL,
)
from src.models.base.application import aplicacion

PARTICION_Q_LABEL = "ParticionQ"
PARTICION_Q_TAG   = "particion_q_strategy"
UMBRAL_EXHAUSTIVO: int = 7  # S(7,5)=140 max → exhaustivo y rápido; greedy para M>7


class ParticionadorQ(SIA):
    """
    Algoritmo greedy secuencial para k-particiones (k ≥ 2).

    Estrategia: Peeling Greedy Secuencial.
    - k-1 fases, en cada fase se extrae un grupo usando el criterio
      submodular (emd_union - emd_delta) sobre los vértices restantes.
    - El último grupo es el conjunto de vértices no asignados.
    - La evaluación usa System.kpartir(), que construye la distribución
      reconstruida como producto tensorial de las marginales de cada grupo.
    """

    def __init__(self, tpm: np.ndarray) -> None:
        super().__init__(tpm)
        gestor_perfilado.start_session(
            f"{NET_LABEL}{len(tpm[COLS_IDX])}{aplicacion.pagina_red_muestra}"
        )
        self._k: int = 2
        self._m: int = 0
        self._n: int = 0
        self._vertices: set[tuple]
        self._indices_futuros: NDArray[np.int8]
        self._indices_presentes: NDArray[np.int8]

        self._cache_delta: dict[tuple, tuple[float, NDArray]] = {}
        self._cache_grupo: dict[tuple, tuple[float, NDArray]] = {}

        self._logger = SafeLogger(PARTICION_Q_TAG)

    # --------------------------------------------------------------
    # Punto de entrada principal
    # --------------------------------------------------------------
    def aplicar_estrategia(
        self,
        estado_inicial: str,
        condicion: str,
        alcance: str,
        mecanismo: str,
        k: int = 2,
    ) -> Solution:
        if k < 2:
            raise ValueError(f"k debe ser ≥ 2. Recibido: k={k}")

        self._k = k
        self.sia_preparar_subsistema(estado_inicial, condicion, alcance, mecanismo)

        if k > self.sia_subsistema.indices_ncubos.size:
            raise ValueError(
                f"k={k} supera las variables futuras ({self.sia_subsistema.indices_ncubos.size})"
            )

        # Construir vértices (tiempo, índice)
        futuros = [(EFFECT, int(i)) for i in self.sia_subsistema.indices_ncubos]
        presentes = [(ACTUAL, int(i)) for i in self.sia_subsistema.dims_ncubos]

        self._m = self.sia_subsistema.indices_ncubos.size
        self._n = self.sia_subsistema.dims_ncubos.size
        self._indices_futuros = self.sia_subsistema.indices_ncubos
        self._indices_presentes = self.sia_subsistema.dims_ncubos
        self._vertices = set(presentes + futuros)

        lista_vertices = presentes + futuros
        grupos, perdida, dist = self._algoritmo_principal(lista_vertices, k)

        texto_particion = self._a_texto(grupos)

        return Solution(
            estrategia=PARTICION_Q_LABEL,
            perdida=perdida,
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=dist,
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=texto_particion,
        )

    # --------------------------------------------------------------
    # Algoritmo principal (k-1 fases de peeling)
    # --------------------------------------------------------------
    @profile(context={TYPE_TAG: PARTICION_Q_TAG})
    def _algoritmo_principal(
        self, vertices: List[Tuple[int, int]], k: int
    ) -> Tuple[List[tuple], float, NDArray]:
        """
        Para M ≤ UMBRAL_EXHAUSTIVO: evalúa TODAS las k-particiones de los futuros
        (número de Stirling S(M,k)) y elige la de menor φ — óptimo global.
        Para M > UMBRAL_EXHAUSTIVO: ejecuta k-1 fases de peeling greedy.
        Retorna (lista_de_grupos, perdida_final, distribucion_final).
        """
        self._logger.critic(
            f"[ParticionadorQ] Iniciado | k={k} | M={self._m} futuros | N={self._n} presentes"
        )

        if self._m <= UMBRAL_EXHAUSTIVO:
            self._logger.critic(
                f"[ParticionadorQ] → Modo EXHAUSTIVO (M={self._m} ≤ {UMBRAL_EXHAUSTIVO})"
            )
            return self._algoritmo_exhaustivo(k)

        # ── Modo greedy peeling ──────────────────────────────────────────────
        self._logger.critic(
            f"[ParticionadorQ] → Modo GREEDY (M={self._m} > {UMBRAL_EXHAUSTIVO})"
        )
        fijos: List[tuple] = []
        restantes = list(vertices)

        for fase in range(k - 1):
            self._logger.critic(f"  Fase {fase+1}/{k-1} | restantes={len(restantes)}")

            if len(restantes) <= 1:
                fijos.append(tuple(restantes))
                restantes = []
                break

            max_tam = self._calcular_maximo_tam_grupo(len(restantes), fase, k)
            mejor = self._busqueda_greedy(restantes, fijos, max_tam)
            fijos.append(mejor)

            usados = set(self._aplanar(mejor))
            restantes = [v for v in restantes if v not in usados]

        if restantes:
            fijos.append(tuple(restantes))

        perdida, dist = self._evaluar_particion_completa(fijos)
        self._logger.critic(f"[ParticionadorQ] Terminado | φ={perdida:.6f} | grupos={len(fijos)}")
        return fijos, perdida, dist

    def _calcular_maximo_tam_grupo(self, num_restantes: int, fase_actual: int, k: int) -> int:
        """Asegura que queden vértices suficientes para las fases siguientes."""
        fases_restantes = (k - 1) - fase_actual
        return max(1, num_restantes - fases_restantes)

    # --------------------------------------------------------------
    # Búsqueda greedy (selección del mejor grupo en una fase)
    # --------------------------------------------------------------
    def _busqueda_greedy(
        self, restantes: List[Tuple[int, int]], fijos: List[tuple], max_tam: int
    ) -> tuple:
        """
        Construye el grupo G_j para la fase actual.
        Sigue el patrón omega/delta del QNodes original.
        """
        # Semilla: priorizar vértices futuros (EFFECT)
        ordenados = sorted(restantes, key=lambda v: (v[0] != EFFECT, v))
        omega = [ordenados[0]]
        delta = ordenados[1:]

        emd_mejor = INFTY_POS
        dist_mejor = None

        limite = len(delta) - 1
        if max_tam > 0:
            limite = min(limite, max_tam - 1)  # -1 porque omega ya tiene la semilla

        for _ in range(limite):
            mejor_local = INFTY_POS
            idx_elegido = 0

            # Recorremos delta (invertimos el orden para cambiar el barrido)
            for i in range(len(delta) - 1, -1, -1):
                emd_union, emd_delta_i, dist_i = self._ganancia_submodular(
                    delta=delta[i],
                    omega=omega,
                    fijos=fijos,
                    todos_restantes=restantes,
                )
                emd_iter = emd_union - emd_delta_i

                if emd_iter < mejor_local:
                    mejor_local = emd_iter
                    idx_elegido = i
                    if emd_delta_i == INT_ZERO:
                        # Partición perfecta, salida inmediata
                        clave = self._a_clave([delta[i]])
                        self._cache_grupo[clave] = (INT_ZERO, dist_i)
                        return (delta[i],)
                    emd_mejor = emd_delta_i
                    dist_mejor = dist_i

            omega.append(delta[idx_elegido])
            delta.pop(idx_elegido)

        clave_grupo = self._a_clave(omega)
        self._cache_grupo[clave_grupo] = (emd_mejor, dist_mejor)
        return tuple(omega)

    # --------------------------------------------------------------
    # Función submodular (costo de agregar delta a omega)
    # --------------------------------------------------------------
    def _ganancia_submodular(
        self,
        delta: Tuple[int, int],
        omega: List,
        fijos: List[tuple],
        todos_restantes: List[Tuple[int, int]],
    ) -> Tuple[float, float, NDArray]:
        """
        Calcula emd_union y emd_delta usando kpartir.
        Retorna (emd_union, emd_delta, distribucion_delta).
        """
        # --- Evaluación individual de delta (con caché) ---
        clave_delta = self._a_clave([delta])
        if clave_delta not in self._cache_delta:
            grupos_delta = self._armar_particion(
                fijos=fijos,
                candidato=[delta],
                resto=todos_restantes,
            )
            part_delta = self.sia_subsistema.kpartir(grupos_delta)
            dist_delta = part_delta.distribucion_marginal()
            emd_delta = emd_efecto(dist_delta, self.sia_dists_marginales)
            self._cache_delta[clave_delta] = (emd_delta, dist_delta)
        else:
            emd_delta, dist_delta = self._cache_delta[clave_delta]

        # --- Evaluación de delta + omega ---
        candidato = [delta] + list(omega)
        grupos_union = self._armar_particion(
            fijos=fijos,
            candidato=candidato,
            resto=todos_restantes,
        )
        part_union = self.sia_subsistema.kpartir(grupos_union)
        dist_union = part_union.distribucion_marginal()
        emd_union = emd_efecto(dist_union, self.sia_dists_marginales)

        return emd_union, emd_delta, dist_delta

    # --------------------------------------------------------------
    # Construcción de la partición para kpartir()
    # --------------------------------------------------------------
    def _armar_particion(
        self,
        fijos: List[tuple],
        candidato: List[Tuple[int, int]],
        resto: List[Tuple[int, int]],
    ) -> List[Tuple[NDArray[np.int8], NDArray[np.int8]]]:
        """
        Convierte grupos de vértices en el formato (F_j, M_j) para kpartir().
        La partición siempre es completa: fijos + candidato + cola.
        """
        # Vértices ya usados en fijos
        usados = set()
        for g in fijos:
            usados.update(self._aplanar(g))
        usados_candidato = set(self._aplanar(candidato))
        usados.update(usados_candidato)

        # Cola = resto - usados
        cola = [v for v in resto if v not in usados]

        resultado = []

        # 1) Grupos fijos
        for g in fijos:
            par = self._a_fj_mj(self._aplanar(g))
            if par is not None:
                resultado.append(par)

        # 2) Grupo candidato
        par_cand = self._a_fj_mj(list(usados_candidato))
        if par_cand is not None:
            resultado.append(par_cand)

        # 3) Cola
        if cola:
            par_cola = self._a_fj_mj(cola)
            if par_cola is not None:
                resultado.append(par_cola)

        return resultado

    # --------------------------------------------------------------
    # Búsqueda exhaustiva por número de Stirling (M ≤ UMBRAL_EXHAUSTIVO)
    # --------------------------------------------------------------
    def _algoritmo_exhaustivo(self, k: int) -> Tuple[List[tuple], float, NDArray]:
        """
        Evalúa TODAS las k-particiones de las M variables futuras (S(M,k) de Stirling).
        Para cada partición de futuros, asigna las N presentes por proximidad de índice
        y evalúa φ con kpartir(). Retorna la tripleta de menor φ.
        """
        pos_futuros = list(range(self._m))
        pos_presentes = list(range(self._n))

        todas_particiones = self._generar_k_particiones_indices(pos_futuros, k)
        self._logger.critic(
            f"[Exhaustivo] S({self._m},{k}) = {len(todas_particiones)} candidatos a evaluar"
        )

        mejor_perdida: float = INFTY_POS
        mejor_dist: Optional[NDArray] = None
        mejor_grupos: Optional[List[tuple]] = None

        for particion_futuros in todas_particiones:
            presentes_por_grupo = self._asignar_presentes_a_grupos(
                particion_futuros, pos_presentes
            )
            # Convertir a formato (F_j, M_j) para kpartir()
            grupos_kpartir = []
            for j in range(k):
                fut_j = np.array(
                    [int(self._indices_futuros[p]) for p in particion_futuros[j]],
                    dtype=np.int8,
                )
                pres_j = np.array(
                    [int(self._indices_presentes[p]) for p in presentes_por_grupo[j]],
                    dtype=np.int8,
                )
                grupos_kpartir.append((fut_j, pres_j))

            try:
                dist = self.sia_subsistema.kpartir(grupos_kpartir).distribucion_marginal()
                perdida = emd_efecto(dist, self.sia_dists_marginales)
            except Exception as e:
                self._logger.critic(f"  Candidato omitido: {e}")
                continue

            if perdida < mejor_perdida:
                mejor_perdida = perdida
                mejor_dist = dist
                mejor_grupos = [
                    tuple(
                        [(EFFECT, int(self._indices_futuros[p])) for p in particion_futuros[j]]
                        + [(ACTUAL, int(self._indices_presentes[p])) for p in presentes_por_grupo[j]]
                    )
                    for j in range(k)
                ]

        if mejor_grupos is None:
            raise RuntimeError(
                f"[Exhaustivo] Sin candidatos válidos para k={k}, M={self._m}"
            )

        self._logger.critic(f"[Exhaustivo] Terminado | φ_min={mejor_perdida:.6f}")
        return mejor_grupos, mejor_perdida, mejor_dist  # type: ignore[return-value]

    def _generar_k_particiones_indices(
        self, indices: List[int], k: int
    ) -> List[List[List[int]]]:
        """
        Genera todas las k-particiones de 'indices' usando el número de Stirling S(n,k).
        Algoritmo recursivo: S(n,k) = k·S(n-1,k) + S(n-1,k-1).
        """
        if k == 1:
            return [[list(indices)]]
        if len(indices) == k:
            return [[[x] for x in indices]]
        if len(indices) < k or k < 1:
            return []
        primero = indices[0]
        resto = indices[1:]
        resultado: List[List[List[int]]] = []
        # Caso A: primer elemento forma su propio grupo singleton
        for p in self._generar_k_particiones_indices(resto, k - 1):
            resultado.append([[primero]] + p)
        # Caso B: primer elemento se une a un grupo existente
        for p in self._generar_k_particiones_indices(resto, k):
            for i in range(len(p)):
                copia = [list(g) for g in p]
                copia[i].append(primero)
                resultado.append(copia)
        return resultado

    def _asignar_presentes_a_grupos(
        self, particion_futuros: List[List[int]], pos_presentes: List[int]
    ) -> List[List[int]]:
        """
        Asigna cada variable presente (por posición) al grupo de futuros
        cuyo índice promedio sea más cercano (distancia de posición).
        """
        k = len(particion_futuros)
        presentes_por_grupo: List[List[int]] = [[] for _ in range(k)]
        promedios = [
            float(np.mean(g)) if g else 0.0 for g in particion_futuros
        ]
        for p in pos_presentes:
            distancias = [abs(p - prom) for prom in promedios]
            j_asignado = int(np.argmin(distancias))
            presentes_por_grupo[j_asignado].append(p)
        return presentes_por_grupo

    # --------------------------------------------------------------
    # Evaluación final de la partición completa
    # --------------------------------------------------------------
    def _evaluar_particion_completa(
        self, grupos: List[tuple]
    ) -> Tuple[float, NDArray]:
        """Calcula EMD de la partición final usando kpartir()."""
        lista_kpartir = []
        for g in grupos:
            par = self._a_fj_mj(self._aplanar(g))
            if par is not None:
                lista_kpartir.append(par)
        sistema_partido = self.sia_subsistema.kpartir(lista_kpartir)
        dist = sistema_partido.distribucion_marginal()
        perdida = emd_efecto(dist, self.sia_dists_marginales)
        return perdida, dist

    # --------------------------------------------------------------
    # Utilidades
    # --------------------------------------------------------------
    def _a_fj_mj(
        self, vertices_planos: List[Tuple[int, int]]
    ) -> Optional[Tuple[NDArray[np.int8], NDArray[np.int8]]]:
        futuros = sorted([idx for t, idx in vertices_planos if t == EFFECT])
        presentes = sorted([idx for t, idx in vertices_planos if t == ACTUAL])
        if not futuros:
            return None
        return (np.array(futuros, dtype=np.int8), np.array(presentes, dtype=np.int8))

    def _aplanar(self, grupo) -> List[Tuple[int, int]]:
        """Convierte cualquier anidamiento en lista plana de tuplas (tiempo, índice)."""
        resultado = []
        for item in grupo:
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[0], (int, np.integer))
                and isinstance(item[1], (int, np.integer))
            ):
                resultado.append((int(item[0]), int(item[1])))
            elif isinstance(item, (list, tuple)):
                resultado.extend(self._aplanar(item))
        return resultado

    def _a_clave(self, vertices: List) -> tuple:
        """Clave canónica ordenada para memoización."""
        return tuple(sorted(self._aplanar(vertices)))

    def _a_texto(self, grupos: List[tuple]) -> str:
        """Formato legible: G1:{FUT|pres}  G2:{...} ..."""
        partes = []
        for i, g in enumerate(grupos, 1):
            flat = self._aplanar(g)
            fut = sorted([v[1] for v in flat if v[0] == EFFECT])
            pres = sorted([v[1] for v in flat if v[0] == ACTUAL])
            fut_str = ",".join(ABECEDARY[idx].upper() for idx in fut) if fut else "-"
            pres_str = ",".join(ABECEDARY[idx].lower() for idx in pres) if pres else "-"
            partes.append(f"G{i}:{{{fut_str}|{pres_str}}}")
        return "  ".join(partes)