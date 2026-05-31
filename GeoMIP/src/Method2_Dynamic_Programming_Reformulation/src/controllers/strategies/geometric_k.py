"""
k_geometric.py — KGeoMIP correcto.

Para k=2: delega completamente a GeometricSIA original.
Para k>=3: reutiliza la tabla T del GeometricSIA original
           y aplica heurísticas geométricas sobre ella.
"""

import time
import numpy as np
from itertools import combinations
from typing import List

from src.controllers.strategies.geometric import GeometricSIA
from src.funcs.base import emd_efecto, ABECEDARY
from src.constants.base import ACTUAL, EFECTO
from src.models.core.solution import Solution

KGEOMIP_LABEL = "KGeoMIP"


class GeometricSIAK(GeometricSIA):
    """
    KGeoMIP: extensión de GeometricSIA para k-particiones (k >= 2).

    Para k=2: usa exactamente el mismo algoritmo que GeometricSIA original.
    Para k>=3: reutiliza la tabla T construida por GeometricSIA y genera
               candidatos de k-partición mediante heurísticas geométricas.
    """

    def __init__(self, gestor):
        super().__init__(gestor)
        self.k: int = 3
        self._candidatos_evaluados: dict = {}

    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
        k: int = 3,
    ):
        """
        Punto de entrada. Para k=2 delega al padre (GeometricSIA original).
        Para k>=3 usa find_mip_k().
        """
        if k < 2:
            raise ValueError(f"k debe ser >= 2. Recibido: k={k}")

        self.k = k

        if k == 2:
            # Delegar completamente al GeometricSIA original — resultados idénticos
            return super().aplicar_estrategia(condicion, alcance, mecanismo, tpm)

        # ── k >= 3 ────────────────────────────────────────────────────────
        n_futuras = self._preparar_sistema(condicion, alcance, mecanismo, tpm)

        if k > n_futuras:
            raise ValueError(
                f"k={k} supera las variables futuras del subsistema ({n_futuras})."
            )

        mip = self.find_mip_k()
        fmt_mip = self._formatear_k_particion_geo(mip)

        return Solution(
            estrategia=KGEOMIP_LABEL,
            perdida=self.memoria_particiones[mip][0],
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=self.memoria_particiones[mip][1],
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    def _preparar_sistema(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
    ) -> int:
        """
        Prepara el subsistema igual que GeometricSIA.aplicar_estrategia().
        Retorna el número de variables futuras del subsistema.
        """
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        futuro = tuple(
            (EFECTO, int(e)) for e in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, int(a)) for a in self.sia_subsistema.dims_ncubos
        )
        self.vertices = set(presente + futuro)

        self._flat_data = [
            ncubo.data.ravel() for ncubo in self.sia_subsistema.ncubos
        ]

        dims = list(self.sia_subsistema.dims_ncubos)
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final = 1 - self.estado_inicial

        return self.sia_subsistema.indices_ncubos.size
    def _ordenar_candidatos_por_costo(self, candidatos: list, costos_fin: list) -> list:
        """
        Ordena candidatos priorizando los que tienen grupos más diferenciados
        entre sí según los costos de la tabla T.
        
        Mayor varianza entre promedios de grupos = grupos más distintos geométricamente
        = más probable que sea una buena partición.
        """
        def score(cand):
            promedios = []
            for _, futuros in cand:
                vals = [costos_fin[i] for i in futuros if i < len(costos_fin)]
                promedios.append(np.mean(vals) if vals else 0.0)
            # Mayor varianza es mejor → negamos para ordenar ascendentemente
            return -np.var(promedios) if len(promedios) > 1 else 0.0

        return sorted(candidatos, key=score)

    #_____________________________________________________________________________________________________
    def find_mip_k(self, max_candidatos: int = 50) -> tuple:
        """
        Encuentra la k-MIP para k>=3.
        Para M<=5: evalúa todos los candidatos (exhaustivo + heurístico).
        Para M>5:  usa heurística con límite de max_candidatos.
        """
        self.sia_logger.critic("Construyendo tabla T para k-partición...")
        self.idx_ncubos = list(range(len(self.sia_subsistema.indices_ncubos)))
        self.caminos = {0: [self.estado_inicial.tolist()]}
        self.tabla_transiciones = {}
        key0 = (tuple(self.caminos[0][0]), tuple(self.caminos[0][0]))
        self.tabla_transiciones[key0] = [0.0] * len(self.idx_ncubos)

        for nivel in range(1, len(self.estado_inicial) + 1):
            self.calcular_costos_nivel(self.estado_final, nivel)
        self.sia_logger.critic("Tabla T construida.")

        M = len(self.idx_ncubos)
        candidatos_k = self._identificar_k_particiones()
        self.sia_logger.critic(f"Candidatos generados: {len(candidatos_k)}")

        # Para M<=5: no recortar — evaluar todos para garantizar optimalidad
        # Para M>5:  recortar a max_candidatos ordenando por heurística
        limite = 500 if M <= 5 else max_candidatos

        if len(candidatos_k) > limite:
            costos_fin = self.tabla_transiciones.get(
                (tuple(self.caminos[0][0]), tuple(self.estado_final)), []
            )
            candidatos_k = self._ordenar_candidatos_por_costo(candidatos_k, costos_fin)
            candidatos_k = candidatos_k[:limite]
            self.sia_logger.critic(
                f"Evaluando los {len(candidatos_k)} candidatos "
                f"{'(exhaustivo)' if M <= 5 else '(heurístico)'}."
            )

        self.memoria_particiones = {}
        self._candidatos_evaluados = {}
        indices_futuros_reales   = list(self.sia_subsistema.indices_ncubos)
        indices_presentes_reales = list(self.sia_subsistema.dims_ncubos)

        for i, cand in enumerate(candidatos_k, 1):
            try:
                grupos_kpartir = self._candidato_a_grupos_kpartir(
                    cand, indices_futuros_reales, indices_presentes_reales
                )
                dist  = self.sia_subsistema.kpartir(grupos_kpartir).distribucion_marginal()
                emd   = emd_efecto(dist, self.sia_dists_marginales)
                clave = self._candidato_a_clave(
                    cand, indices_futuros_reales, indices_presentes_reales
                )
                self.memoria_particiones[clave] = (emd, dist)
                self._candidatos_evaluados[clave] = cand
                self.sia_logger.critic(
                    f"Candidato {i}/{len(candidatos_k)}: φ={emd:.6f}"
                )
            except Exception as e:
                self.sia_logger.critic(f"Candidato {i} omitido: {e}")
                continue

        if not self.memoria_particiones:
            raise RuntimeError("No se generaron candidatos válidos para k>=3.")

        mejor = min(
            self.memoria_particiones,
            key=lambda c: self.memoria_particiones[c][0]
        )
        self.sia_logger.critic(
            f"Mejor pérdida: {self.memoria_particiones[mejor][0]:.6f} "
            f"de {len(self.memoria_particiones)} candidatos únicos evaluados."
        )
        return mejor


    def _identificar_k_particiones(self) -> List[List[List[int]]]:
        """
        Genera candidatos de k-partición para k>=3.

        Para M<=5: añade todas las k-particiones exactas (garantiza optimalidad).
        Para M>5:  usa solo heurísticas (eficiencia en sistemas grandes).

        Cada candidato es una lista de k pares [pos_presentes_j, pos_futuros_j]
        con POSICIONES en indices_ncubos/dims_ncubos (no índices reales).
        """
        k = self.k
        s0    = self.caminos[0][0]
        s_fin = self.estado_final.tolist()
        clave_fin  = (tuple(s0), tuple(s_fin))
        costos_fin = self.tabla_transiciones.get(clave_fin, [])

        M = len(self.idx_ncubos)     # variables futuras
        N = len(self.estado_inicial) # variables presentes

        if not costos_fin or M < k:
            return []

        candidatos: List[List[List[int]]] = []

        # ── Capa 0: Exhaustiva para sistemas pequeños (M<=5) ──────────────────
        # Garantiza que la partición óptima siempre esté entre los candidatos.
        if M <= 5:
            todas_particiones = self._generar_k_particiones_indices(list(range(M)), k)
            for particion in todas_particiones:
                # Para la partición exacta, los presentes son los mismos índices que los futuros
                cand = [[list(grupo), list(grupo)] for grupo in particion]
                if cand not in candidatos:
                    candidatos.append(cand)

            self.sia_logger.critic(
                f"Capa 0 exhaustiva: {len(candidatos)} particiones exactas "
                f"para M={M}, k={k} (S({M},{k})={len(todas_particiones)})."
            )

        # ── Capa 1a: División uniforme por cuantiles ───────────────────────────
        orden_pos = sorted(range(M), key=lambda pos: costos_fin[pos])
        chunks    = self._dividir_en_k(orden_pos, k)

        if all(len(c) > 0 for c in chunks):
            presentes_por_grupo = self._asignar_todos_presentes(
                chunks, costos_fin, N
            )
            cand = [
                [presentes_por_grupo[j], list(chunks[j])]
                for j in range(k)
            ]
            if cand not in candidatos:
                candidatos.append(cand)

        # ── Capa 1b: División por gaps naturales ───────────────────────────────
        if M > k:
            costos_ord   = [costos_fin[pos] for pos in orden_pos]
            diffs        = [
                (costos_ord[i+1] - costos_ord[i], i+1)
                for i in range(M - 1)
            ]
            puntos_corte = sorted(
                pos for _, pos in sorted(diffs, reverse=True)[:k-1]
            )
            grupos_gap, prev = [], 0
            for corte in puntos_corte:
                grupos_gap.append([orden_pos[i] for i in range(prev, corte)])
                prev = corte
            grupos_gap.append([orden_pos[i] for i in range(prev, M)])

            if len(grupos_gap) == k and all(len(g) > 0 for g in grupos_gap):
                presentes_por_grupo = self._asignar_todos_presentes(
                    grupos_gap, costos_fin, N
                )
                cand = [
                    [presentes_por_grupo[j], list(grupos_gap[j])]
                    for j in range(k)
                ]
                if cand not in candidatos:
                    candidatos.append(cand)

        # ── Capa 2: k-medoids con estados de referencia ────────────────────────
        mitad      = max(1, N // 2)
        estados_ref: List[tuple] = []
        for nivel in range(1, mitad + 1):
            for estado in self.caminos.get(nivel, []):
                estados_ref.append(tuple(estado))

        max_refs    = min(len(estados_ref), 4 * k)
        estados_ref = estados_ref[:max_refs]

        if len(estados_ref) >= k - 1:
            refs_vistos: set = set()
            for refs in combinations(estados_ref, min(k-1, len(estados_ref))):
                clave_refs = frozenset(refs)
                if clave_refs in refs_vistos:
                    continue
                refs_vistos.add(clave_refs)

                grupos_futuros = [[] for _ in range(k)]
                for pos in range(M):
                    costos_ref_pos = []
                    for ref in refs:
                        c_ref = self.tabla_transiciones.get(
                            (tuple(s0), ref), [float("inf")] * M
                        )
                        costos_ref_pos.append(
                            c_ref[pos] if pos < len(c_ref) else float("inf")
                        )
                    costos_ref_pos.append(
                        costos_fin[pos] if pos < len(costos_fin) else float("inf")
                    )
                    j_star = costos_ref_pos.index(min(costos_ref_pos))
                    grupos_futuros[j_star].append(pos)

                if not all(len(g) > 0 for g in grupos_futuros):
                    continue

                presentes_por_grupo = self._asignar_todos_presentes(
                    grupos_futuros, costos_fin, N
                )
                cand = [
                    [presentes_por_grupo[j], list(grupos_futuros[j])]
                    for j in range(k)
                ]
                if cand not in candidatos:
                    candidatos.append(cand)

        # ── Fallback ───────────────────────────────────────────────────────────
        if not candidatos:
            fallback = self._dividir_en_k(list(range(M)), k)
            presentes_por_grupo = self._asignar_todos_presentes(
                fallback, costos_fin, N
            )
            cand = [
                [presentes_por_grupo[j], list(fallback[j])]
                for j in range(k)
            ]
            candidatos.append(cand)

        return candidatos


    def _generar_k_particiones_indices(
        self,
        indices: List[int],
        k: int,
    ) -> List[List[List[int]]]:
        """
        Genera todas las k-particiones posibles de una lista de índices.
        Usa recursión estilo Stirling — igual que _generar_k_particiones
        de GeometricSIA pero operando sobre índices enteros directamente.

        Args:
            indices: Lista de posiciones (enteros) a particionar.
            k:       Número de grupos deseados.
        Returns:
            Lista de particiones; cada partición es una lista de k grupos
            (cada grupo es una lista de índices).
        """
        if k == 1:
            return [[list(indices)]]
        if len(indices) == k:
            return [[[x] for x in indices]]
        if len(indices) < k or k < 1:
            return []

        primer   = indices[0]
        resto    = indices[1:]
        resultado: List[List[List[int]]] = []

        # Caso A: primer elemento forma su propio grupo
        for p in self._generar_k_particiones_indices(resto, k - 1):
            resultado.append([[primer]] + p)

        # Caso B: primer elemento se une a un grupo existente
        for p in self._generar_k_particiones_indices(resto, k):
            for i in range(len(p)):
                copia = [list(g) for g in p]
                copia[i].append(primer)
                resultado.append(copia)

        return resultado
     # ══════════════════════════════════════════════════════════════════════
    #  MÉTODOS AUXILIARES
    # ══════════════════════════════════════════════════════════════════════

    def _asignar_todos_presentes(
        self,
        grupos_futuros_pos: List[List[int]],
        costos_fin: list,
        N: int,
    ) -> List[List[int]]:
        """
        Asigna posiciones de variables presentes a grupos de forma disjunta.

        Cada posición presente i se asigna al grupo j cuyo promedio de costos
        de futuros es más cercano al costo de i en la tabla T.

        Garantía: cada presente va a exactamente un grupo.

        Args:
            grupos_futuros_pos: k listas de posiciones de futuros.
            costos_fin: vector de costos tx(s0, s_final) de la tabla T.
            N: número de variables presentes.
        Returns:
            k listas de posiciones de presentes.
        """
        k = len(grupos_futuros_pos)
        presentes_por_grupo = [[] for _ in range(k)]

        # Costo promedio de cada grupo de futuros
        costos_promedio = []
        for grupo in grupos_futuros_pos:
            if grupo:
                vals = [costos_fin[p] for p in grupo if p < len(costos_fin)]
                promedio = np.mean(vals) if vals else 0.0
            else:
                promedio = 0.0
            costos_promedio.append(promedio)

        # Asignar cada posición presente al grupo más cercano
        for i in range(N):
            costo_i = costos_fin[i] if i < len(costos_fin) else 0.0
            distancias = [abs(costo_i - prom) for prom in costos_promedio]
            j_asignado = distancias.index(min(distancias))
            presentes_por_grupo[j_asignado].append(i)

        # Garantía: grupos vacíos reciben presentes sobrantes
        libres = []
        for i in range(N):
            if not any(i in g for g in presentes_por_grupo):
                libres.append(i)

        for j in range(k):
            if not presentes_por_grupo[j] and libres:
                presentes_por_grupo[j].append(libres.pop(0))

        return presentes_por_grupo

    def _dividir_en_k(self, lista: list, k: int) -> List[list]:
        """Divide lista en k sublistas de tamaño uniforme."""
        n = len(lista)
        base, extra = divmod(n, k)
        grupos = []
        inicio = 0
        for j in range(k):
            tam = base + (1 if j < extra else 0)
            grupos.append(lista[inicio:inicio + tam])
            inicio += tam
        return grupos

    def _candidato_a_grupos_kpartir(
        self,
        candidato: List[List[int]],
        indices_futuros_reales: list,
        indices_presentes_reales: list,
    ) -> List[tuple]:
        """
        Convierte candidato de posiciones a formato (F_j, M_j) para kpartir().

        Los candidatos usan POSICIONES (0, 1, 2, ...) en los arrays de índices.
        kpartir() necesita los ÍNDICES REALES de los NCubos y dims.

        Args:
            candidato: k pares [pos_pres_j, pos_fut_j].
            indices_futuros_reales: array de índices reales de NCubos.
            indices_presentes_reales: array de índices reales de dims.
        Returns:
            Lista de k pares (F_j_real, M_j_real) para kpartir().
        Raises:
            ValueError: Si algún grupo futuro está vacío.
        """
        grupos = []
        for pres_j, fut_j in candidato:
            if not fut_j:
                raise ValueError("Grupo futuro vacío")
            # Convertir a enteros explícitamente y luego a array de int8
            fut_reales = [int(indices_futuros_reales[p]) for p in fut_j]
            pres_reales = [int(indices_presentes_reales[p]) for p in pres_j] if pres_j else []
            F_j = np.array(fut_reales, dtype=np.int8)
            M_j = np.array(pres_reales, dtype=np.int8) if pres_reales else np.array([], dtype=np.int8)
            grupos.append((F_j, M_j))
        return grupos

    def _candidato_a_clave(
        self,
        candidato: List[List[int]],
        indices_futuros_reales: list,
        indices_presentes_reales: list,
    ) -> tuple:
        """
        Genera clave hashable que preserva la estructura de agrupación.
        
        Cada bloque se representa como una tupla ordenada de sus vértices,
        y los bloques se ordenan entre sí para canonicalizar.
        Así dos candidatos con diferente agrupación nunca colisionan.
        """
        bloques = []
        for pos_pres_j, pos_fut_j in candidato:
            fut_reales = tuple(sorted(
                int(indices_futuros_reales[p]) for p in pos_fut_j
            ))
            pre_reales = tuple(sorted(
                int(indices_presentes_reales[p]) for p in pos_pres_j
            ))
            # Cada bloque es (futuros_ordenados, presentes_ordenados)
            bloques.append((fut_reales, pre_reales))
        
        # Ordenar los bloques para canonicalizar
        # (el orden de los grupos no importa, solo la agrupación)
        return tuple(sorted(bloques))

    def _formatear_k_particion_geo(self, clave: tuple) -> str:
        """
        Formatea la k-partición mostrando los k grupos por separado.
        clave = ((fut_reales_1, pre_reales_1), (fut_reales_2, pre_reales_2), ...)
        """
        if (not hasattr(self, '_candidatos_evaluados')
                or clave not in self._candidatos_evaluados):
            # Fallback usando la nueva estructura de clave
            partes = []
            for j, (fut_reales, pre_reales) in enumerate(clave):
                fut_str = ",".join(ABECEDARY[f].upper() for f in fut_reales) or "-"
                pre_str = ",".join(ABECEDARY[p].lower() for p in pre_reales) or "-"
                partes.append(f"G{j+1}:{{{fut_str}|{pre_str}}}")
            return f"k={self.k} | " + " | ".join(partes)

        indices_futuros_reales   = list(self.sia_subsistema.indices_ncubos)
        indices_presentes_reales = list(self.sia_subsistema.dims_ncubos)
        candidato = self._candidatos_evaluados[clave]

        partes = []
        for j, (pos_pres_j, pos_fut_j) in enumerate(candidato):
            fut_reales = [int(indices_futuros_reales[p]) for p in pos_fut_j]
            pre_reales = [int(indices_presentes_reales[p]) for p in pos_pres_j]
            fut_str = ",".join(ABECEDARY[f].upper() for f in fut_reales) or "-"
            pre_str = ",".join(ABECEDARY[p].lower() for p in pre_reales) or "-"
            partes.append(f"G{j+1}:{{{fut_str}|{pre_str}}}")

        return f"k={self.k} | " + " | ".join(partes)

# Alias canónico
KGeoMIP = GeometricSIAK