"""
k_geometric.py — KGeoMIP corregido.

Para k=2: delega completamente a GeometricSIA original (resultados idénticos).
Para k>=3: reutiliza la tabla T construida una sola vez por subsistema
           y aplica heurísticas geométricas sobre ella.

CORRECCIONES APLICADAS:
1. _asignar_todos_presentes: ahora respeta la CORRESPONDENCIA POR VARIABLE
   FÍSICA entre presentes y futuros (si X^{t+1} está en el grupo j, X^t
   también va al grupo j). La versión anterior asignaba por "cercanía de
   costo promedio", lo cual separaba X^t de X^{t+1} y rompía la
   dependencia causal, inflando el EMD en órdenes de magnitud (φ(k=3)
   pasaba de ~0.47 a ~4.6 en los casos de prueba).

2. Nueva capa de generación de candidatos por BISECCIÓN RECURSIVA: parte
   de un único grupo con todas las posiciones futuras y lo divide
   sucesivamente en 2 (el grupo con mayor dispersión de costos) hasta
   llegar a k grupos. Esto produce particiones que son refinamientos
   anidados —análogos al "peeling" de QNodes— manteniendo φ(k) cercano
   a φ(k-1) en vez de saltos abruptos.
"""

import time
import numpy as np
from itertools import combinations
from typing import List, Tuple, Dict, Any, Optional

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
        self._candidatos_evaluados: Dict[tuple, Any] = {}
        self._tabla_construida: bool = False  # evita reconstruir tabla T

    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
        tpm: np.ndarray,
        k: int = 3,
    ) -> Solution:
        """
        Punto de entrada. Para k=2 delega al padre (GeometricSIA original).
        Para k>=3 construye la tabla T una sola vez y busca la k-MIP.

        Args:
            condicion: cadena con índices de variables de condición.
            alcance: cadena con índices de variables de alcance.
            mecanismo: cadena con índices de variables del mecanismo.
            tpm: matriz de probabilidades de transición.
            k: número de partes (2 ≤ k ≤ 5).

        Returns:
            Solution con la partición óptima encontrada.
        """
        if k < 2:
            raise ValueError(f"k debe ser >= 2. Recibido: k={k}")

        self.k = k

        # Caso k=2: delegación total al padre
        if k == 2:
            return super().aplicar_estrategia(condicion, alcance, mecanismo, tpm)

        # ── k >= 3 ────────────────────────────────────────────────────────
        # Preparar subsistema (reutiliza la lógica del padre sin sobrescribir)
        self._preparar_subsistema_padre(condicion, alcance, mecanismo, tpm)

        n_futuras = len(self.sia_subsistema.indices_ncubos)
        if k > n_futuras:
            raise ValueError(
                f"k={k} supera las variables futuras del subsistema ({n_futuras})."
            )

        # Construir tabla T solo si no se ha construido ya para este subsistema
        if not self._tabla_construida:
            self._construir_tabla_T()
            self._tabla_construida = True

        mip = self._find_mip_k()
        fmt_mip = self._formatear_k_particion_geo(mip)

        return Solution(
            estrategia=KGEOMIP_LABEL,
            perdida=self.memoria_particiones[mip][0],
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=self.memoria_particiones[mip][1],
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )

    def _preparar_subsistema_padre(
        self, condicion: str, alcance: str, mecanismo: str, tpm: np.ndarray
    ) -> None:
        """
        Prepara el subsistema usando exactamente el mismo código que
        GeometricSIA.aplicar_estrategia(), pero sin delegar completamente
        porque necesitamos mantener el contexto para k>=3.
        """
        self.sia_preparar_subsistema(condicion, alcance, mecanismo, tpm)

        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final = 1 - self.estado_inicial

        futuro = tuple((EFECTO, int(e)) for e in self.sia_subsistema.indices_ncubos)
        presente = tuple((ACTUAL, int(a)) for a in self.sia_subsistema.dims_ncubos)
        self.vertices = set(presente + futuro)

        self._flat_data = [ncubo.data.ravel() for ncubo in self.sia_subsistema.ncubos]

    def _construir_tabla_T(self) -> None:
        """
        Construye la tabla de costos de transiciones (tabla T) una sola vez
        y la almacena en self.tabla_transiciones.
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

    # -------------------------------------------------------------
    #  Métodos principales de búsqueda de k-MIP
    # -------------------------------------------------------------
    def _find_mip_k(self, max_candidatos: int = 50) -> tuple:
        """
        Encuentra la k-MIP para k>=3.
        Para M<=5: evalúa TODAS las k-particiones exactas (óptimo global).
        Para M>5:  usa heurística con límite de max_candidatos.

        Args:
            max_candidatos: número máximo de candidatos a evaluar (solo para M>5).

        Returns:
            Clave de la mejor partición encontrada.
        """
        M = len(self.idx_ncubos)  # número de variables futuras
        candidatos_k = self._identificar_k_particiones(M)

        self.sia_logger.critic(f"Candidatos generados: {len(candidatos_k)}")

        # Para M<=5: evaluamos todos (exhaustivo). Para M>5: recortamos si es necesario.
        if M > 5 and len(candidatos_k) > max_candidatos:
            # Ordenar por heurística y tomar los mejores
            costos_fin = self._obtener_costos_fin()
            candidatos_k = self._ordenar_candidatos_por_costo(candidatos_k, costos_fin)
            candidatos_k = candidatos_k[:max_candidatos]
            self.sia_logger.critic(
                f"Evaluando los {len(candidatos_k)} mejores candidatos (heurístico)."
            )
        else:
            self.sia_logger.critic(
                f"Evaluando todos los {len(candidatos_k)} candidatos (exhaustivo)."
            )

        self.memoria_particiones = {}
        self._candidatos_evaluados = {}
        indices_futuros_reales = list(self.sia_subsistema.indices_ncubos)
        indices_presentes_reales = list(self.sia_subsistema.dims_ncubos)

        for i, cand in enumerate(candidatos_k, 1):
            try:
                grupos_kpartir = self._candidato_a_grupos_kpartir(
                    cand, indices_futuros_reales, indices_presentes_reales
                )
                dist = self.sia_subsistema.kpartir(grupos_kpartir).distribucion_marginal()
                emd = emd_efecto(dist, self.sia_dists_marginales)

                clave = self._candidato_a_clave(
                    cand, indices_futuros_reales, indices_presentes_reales
                )
                self.memoria_particiones[clave] = (emd, dist)
                self._candidatos_evaluados[clave] = cand

                self.sia_logger.critic(f"Candidato {i}/{len(candidatos_k)}: φ={emd:.6f}")

            except Exception as e:
                self.sia_logger.critic(f"Candidato {i} omitido: {e}")
                continue

        if not self.memoria_particiones:
            raise RuntimeError("No se generaron candidatos válidos para k>=3.")

        mejor = min(self.memoria_particiones, key=lambda c: self.memoria_particiones[c][0])
        self.sia_logger.critic(
            f"Mejor pérdida: {self.memoria_particiones[mejor][0]:.6f} "
            f"de {len(self.memoria_particiones)} candidatos únicos evaluados."
        )
        return mejor

    def _obtener_costos_fin(self) -> List[float]:
        """Devuelve el vector de costos desde estado_inicial a estado_final."""
        s0 = self.caminos[0][0]
        s_fin = self.estado_final.tolist()
        clave_fin = (tuple(s0), tuple(s_fin))
        return self.tabla_transiciones.get(clave_fin, [])

    # -------------------------------------------------------------
    #  Generación de candidatos
    # -------------------------------------------------------------
    def _identificar_k_particiones(self, M: int) -> List[List[List[int]]]:
        """
        Genera candidatos de k-partición para k>=3.

        - Capa -1 (NUEVA): bisección recursiva. Genera UNA partición por
          refinamientos sucesivos (split del grupo con mayor dispersión de
          costos), análoga al "peeling" de QNodes. Esto tiende a producir
          φ(k) cercano a φ(k-1) en vez de saltos abruptos.
        - Capa 0: exhaustiva para M<=5.
        - Capas 1a/1b/2: heurísticas (cuantiles, gaps, k-medoids), como antes.

        Cada candidato es una lista de k pares [pos_presentes_j, pos_futuros_j]
        con POSICIONES en indices_ncubos/dims_ncubos (no índices reales).
        """
        k = self.k
        costos_fin = self._obtener_costos_fin()
        if not costos_fin or M < k:
            return []

        candidatos: List[List[List[int]]] = []

        # Capa -1: Bisección recursiva (refinamiento anidado)
        cand_bisec = self._generar_particion_biseccion_recursiva(M, k, costos_fin)
        if cand_bisec is not None and cand_bisec not in candidatos:
            candidatos.append(cand_bisec)

        # Capa 0: Exhaustiva para M<=5
        if M <= 5:
            todas_particiones = self._generar_k_particiones_indices(list(range(M)), k)
            for particion in todas_particiones:
                pres_j = self._asignar_todos_presentes(particion, costos_fin)
                cand = [[pres_j[j], list(particion[j])] for j in range(k)]
                if cand not in candidatos:
                    candidatos.append(cand)

        # Capa 1a: División uniforme por cuantiles
        orden_pos = sorted(range(M), key=lambda pos: costos_fin[pos])
        chunks = self._dividir_en_k(orden_pos, k)
        if all(len(c) > 0 for c in chunks):
            presentes = self._asignar_todos_presentes(chunks, costos_fin)
            cand = [[presentes[j], list(chunks[j])] for j in range(k)]
            if cand not in candidatos:
                candidatos.append(cand)

        # Capa 1b: División por gaps naturales
        if M > k:
            costos_ord = [costos_fin[pos] for pos in orden_pos]
            diffs = [(costos_ord[i+1] - costos_ord[i], i+1) for i in range(M-1)]
            puntos_corte = sorted(pos for _, pos in sorted(diffs, reverse=True)[:k-1])
            grupos_gap, prev = [], 0
            for corte in puntos_corte:
                grupos_gap.append([orden_pos[i] for i in range(prev, corte)])
                prev = corte
            grupos_gap.append([orden_pos[i] for i in range(prev, M)])
            if len(grupos_gap) == k and all(len(g) > 0 for g in grupos_gap):
                presentes = self._asignar_todos_presentes(grupos_gap, costos_fin)
                cand = [[presentes[j], list(grupos_gap[j])] for j in range(k)]
                if cand not in candidatos:
                    candidatos.append(cand)

        # Capa 2: k-medoids con estados de referencia (solo si hay suficientes estados)
        estados_ref = self._obtener_estados_referencia()
        if len(estados_ref) >= k - 1:
            refs_vistos = set()
            max_refs = min(len(estados_ref), 4 * k)
            estados_ref = estados_ref[:max_refs]
            for refs in combinations(estados_ref, min(k-1, len(estados_ref))):
                clave_refs = frozenset(refs)
                if clave_refs in refs_vistos:
                    continue
                refs_vistos.add(clave_refs)

                grupos_futuros = [[] for _ in range(k)]
                s0 = tuple(self.caminos[0][0])
                for pos in range(M):
                    costos_ref_pos = []
                    for ref in refs:
                        c_ref = self.tabla_transiciones.get((s0, ref), [float('inf')] * M)
                        costos_ref_pos.append(c_ref[pos] if pos < len(c_ref) else float('inf'))
                    costos_ref_pos.append(costos_fin[pos] if pos < len(costos_fin) else float('inf'))
                    j_star = costos_ref_pos.index(min(costos_ref_pos))
                    grupos_futuros[j_star].append(pos)

                if not all(len(g) > 0 for g in grupos_futuros):
                    continue

                presentes = self._asignar_todos_presentes(grupos_futuros, costos_fin)
                cand = [[presentes[j], list(grupos_futuros[j])] for j in range(k)]
                if cand not in candidatos:
                    candidatos.append(cand)

        # Fallback: si no hay ningún candidato, dividir uniformemente
        if not candidatos:
            fallback = self._dividir_en_k(list(range(M)), k)
            presentes = self._asignar_todos_presentes(fallback, costos_fin)
            cand = [[presentes[j], list(fallback[j])] for j in range(k)]
            candidatos.append(cand)

        return candidatos

    def _generar_particion_biseccion_recursiva(
        self, M: int, k: int, costos_fin: List[float]
    ) -> Optional[List[List[List[int]]]]:
        """
        Genera UNA k-partición de futuros mediante bisección recursiva:

        1. Inicia con un único grupo {0, 1, ..., M-1}.
        2. Mientras haya menos de k grupos:
           a. Elige el grupo con mayor "dispersión" (max - min de
              costos_fin entre sus elementos) que tenga >= 2 elementos.
           b. Lo ordena por costo y lo divide en dos mitades.
           c. Reemplaza ese grupo por las dos mitades.
        3. Asigna presentes por correspondencia de variable física.

        Esta partición es un REFINAMIENTO ANIDADO: cada paso solo separa
        un subconjunto que antes estaba unido, preservando la estructura
        de los demás grupos. Esto es análogo al peeling secuencial de
        QNodes y favorece que φ(k) quede cercano a φ(k-1) en lugar de
        saltar a un valor muy distinto.

        Returns:
            Candidato [[pres_pos, fut_pos], ...] de tamaño k, o None si
            no se puede construir (p.ej. M < k).
        """
        if M < k:
            return None

        grupos: List[List[int]] = [list(range(M))]

        def dispersion(grupo: List[int]) -> float:
            vals = [costos_fin[p] for p in grupo if p < len(costos_fin)]
            if not vals:
                return 0.0
            return max(vals) - min(vals)

        while len(grupos) < k:
            divisibles = [(i, g) for i, g in enumerate(grupos) if len(g) >= 2]
            if not divisibles:
                # No quedan grupos divisibles (todos tamaño 1) pero faltan grupos
                return None

            idx_split, grupo_split = max(divisibles, key=lambda ig: dispersion(ig[1]))

            orden = sorted(
                grupo_split,
                key=lambda p: costos_fin[p] if p < len(costos_fin) else 0.0,
            )
            mitad = max(1, len(orden) // 2)
            sub1, sub2 = orden[:mitad], orden[mitad:]

            if not sub1 or not sub2:
                return None

            grupos = grupos[:idx_split] + [sub1, sub2] + grupos[idx_split + 1:]

        if len(grupos) != k or any(len(g) == 0 for g in grupos):
            return None

        presentes = self._asignar_todos_presentes(grupos, costos_fin)
        return [[presentes[j], list(grupos[j])] for j in range(k)]

    def _obtener_estados_referencia(self) -> List[tuple]:
        """Extrae estados intermedios de self.caminos para usar como referencias."""
        N = len(self.estado_inicial)
        mitad = max(1, N // 2)
        estados: List[tuple] = []
        for nivel in range(1, mitad + 1):
            for estado in self.caminos.get(nivel, []):
                estados.append(tuple(estado))
        return estados

    # -------------------------------------------------------------
    #  Métodos auxiliares
    # -------------------------------------------------------------
    def _ordenar_candidatos_por_costo(self, candidatos: list, costos_fin: list) -> list:
        """
        Ordena candidatos priorizando aquellos cuyos grupos de futuros tengan
        promedios de costos más diferenciados (mayor varianza entre grupos).
        """
        def score(cand):
            promedios = []
            for _, futuros in cand:
                vals = [costos_fin[i] for i in futuros if i < len(costos_fin)]
                promedios.append(np.mean(vals) if vals else 0.0)
            return -np.var(promedios) if len(promedios) > 1 else 0.0
        return sorted(candidatos, key=score)

    def _generar_k_particiones_indices(self, indices: List[int], k: int) -> List[List[List[int]]]:
        """Genera todas las k-particiones de una lista de índices (Stirling recursivo)."""
        if k == 1:
            return [[list(indices)]]
        if len(indices) == k:
            return [[[x] for x in indices]]
        if len(indices) < k or k < 1:
            return []
        primero = indices[0]
        resto = indices[1:]
        resultado = []
        # Primer elemento en su propio grupo
        for p in self._generar_k_particiones_indices(resto, k - 1):
            resultado.append([[primero]] + p)
        # Primer elemento se une a un grupo existente
        for p in self._generar_k_particiones_indices(resto, k):
            for i in range(len(p)):
                copia = [list(g) for g in p]
                copia[i].append(primero)
                resultado.append(copia)
        return resultado

    def _asignar_todos_presentes(
        self, grupos_futuros: List[List[int]], costos_fin: List[float]
    ) -> List[List[int]]:
        """
        Asigna cada variable presente (posición 0..N-1 en dims_ncubos) al
        grupo que contiene la MISMA variable física en su versión futura.

        CORRECCIÓN CLAVE: si la variable global `v` aparece tanto en
        `indices_ncubos` (futuro, en algún grupo j) como en `dims_ncubos`
        (presente), entonces la posición presente de `v` DEBE ir al MISMO
        grupo j. De lo contrario, kpartir calcula P(X^{t+1} | grupo sin
        X^t), rompiendo la dependencia causal y generando un EMD inflado
        en órdenes de magnitud.

        Para variables "presente-only" (en mecanismo pero no en alcance,
        o viceversa, sin contraparte en los futuros) se usa como fallback
        la heurística original de cercanía de costo promedio del grupo.

        Args:
            grupos_futuros: lista de k listas con POSICIONES de futuros
                            (índices en indices_ncubos).
            costos_fin: vector de costos desde estado_inicial a
                        estado_final, indexado por posición futura.

        Returns:
            Lista de k listas con posiciones (en dims_ncubos) de presentes
            asignadas. Ningún grupo queda vacío (si hay suficientes
            elementos para repartir).
        """
        k = len(grupos_futuros)
        indices_futuros_reales = list(self.sia_subsistema.indices_ncubos)
        indices_presentes_reales = list(self.sia_subsistema.dims_ncubos)
        N = len(indices_presentes_reales)

        # --- Paso 1: variable global -> grupo, según partición de futuros ---
        var_a_grupo: Dict[int, int] = {}
        for j, grupo in enumerate(grupos_futuros):
            for p in grupo:
                if p < len(indices_futuros_reales):
                    var_a_grupo[int(indices_futuros_reales[p])] = j

        presentes_por_grupo: List[List[int]] = [[] for _ in range(k)]

        # --- Paso 2: costo promedio de cada grupo de futuros (fallback) ---
        promedios = []
        for grupo in grupos_futuros:
            if grupo:
                vals = [costos_fin[p] for p in grupo if p < len(costos_fin)]
                prom = np.mean(vals) if vals else 0.0
            else:
                prom = 0.0
            promedios.append(prom)
        mediana = float(np.median(costos_fin)) if costos_fin else 0.0

        # --- Paso 3: asignación por correspondencia directa de variable ---
        sin_asignar: List[int] = []
        for q in range(N):
            var_global = int(indices_presentes_reales[q])
            if var_global in var_a_grupo:
                presentes_por_grupo[var_a_grupo[var_global]].append(q)
            else:
                sin_asignar.append(q)

        # --- Paso 4: fallback heurístico (cercanía de costo) ---
        for q in sin_asignar:
            costo_q = costos_fin[q] if q < len(costos_fin) else mediana
            distancias = [abs(costo_q - prom) for prom in promedios]
            j_asignado = distancias.index(min(distancias))
            presentes_por_grupo[j_asignado].append(q)

        # --- Paso 5: asegurar que ningún grupo quede vacío ---
        for j in range(k):
            if not presentes_por_grupo[j]:
                largest_group = max(presentes_por_grupo, key=len)
                if len(largest_group) > 1:
                    presentes_por_grupo[j].append(largest_group.pop())

        return presentes_por_grupo

    def _dividir_en_k(self, lista: list, k: int) -> List[list]:
        """Divide lista en k sublistas de tamaño lo más uniforme posible."""
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
        Convierte candidato (posiciones) a formato (F_j, M_j) para kpartir().
        """
        grupos = []
        for pres_j, fut_j in candidato:
            if not fut_j:
                raise ValueError("Grupo futuro vacío")
            fut_reales = np.array([indices_futuros_reales[p] for p in fut_j], dtype=np.int8)
            pres_reales = np.array([indices_presentes_reales[p] for p in pres_j], dtype=np.int8) if pres_j else np.array([], dtype=np.int8)
            grupos.append((fut_reales, pres_reales))
        return grupos

    def _candidato_a_clave(
        self,
        candidato: List[List[int]],
        indices_futuros_reales: list,
        indices_presentes_reales: list,
    ) -> tuple:
        """
        Genera clave hashable que preserve la estructura de agrupación.
        """
        bloques = []
        for pos_pres_j, pos_fut_j in candidato:
            fut_reales = tuple(sorted(int(indices_futuros_reales[p]) for p in pos_fut_j))
            pre_reales = tuple(sorted(int(indices_presentes_reales[p]) for p in pos_pres_j))
            bloques.append((fut_reales, pre_reales))
        return tuple(sorted(bloques))

    def _formatear_k_particion_geo(self, clave: tuple) -> str:
        """
        Formatea la k-partición mostrando los k grupos.
        """
        if not hasattr(self, '_candidatos_evaluados') or clave not in self._candidatos_evaluados:
            # Fallback: usar clave directamente
            partes = []
            for j, (fut_reales, pre_reales) in enumerate(clave):
                fut_str = ",".join(ABECEDARY[f].upper() for f in fut_reales) or "-"
                pre_str = ",".join(ABECEDARY[p].lower() for p in pre_reales) or "-"
                partes.append(f"G{j+1}:{{{fut_str}|{pre_str}}}")
            return f"k={self.k} | " + " | ".join(partes)

        indices_futuros_reales = list(self.sia_subsistema.indices_ncubos)
        indices_presentes_reales = list(self.sia_subsistema.dims_ncubos)
        candidato = self._candidatos_evaluados[clave]

        partes = []
        for j, (pos_pres_j, pos_fut_j) in enumerate(candidato):
            fut_reales = [indices_futuros_reales[p] for p in pos_fut_j]
            pre_reales = [indices_presentes_reales[p] for p in pos_pres_j]
            fut_str = ",".join(ABECEDARY[f].upper() for f in fut_reales) or "-"
            pre_str = ",".join(ABECEDARY[p].lower() for p in pre_reales) or "-"
            partes.append(f"G{j+1}:{{{fut_str}|{pre_str}}}")
        return f"k={self.k} | " + " | ".join(partes)

    def evaluar_distribucion_k_particion(self, particion: list) -> np.ndarray:
        """
        Calcula la distribución marginal reconstruida para una partición dada
        en el formato de variables [(tipo, variable), ...].
        """
        grupos_kpartir = []
        for bloque in particion:
            fut = np.array([v[1] for v in bloque if v[0] == EFECTO], dtype=np.int8)
            pres = np.array([v[1] for v in bloque if v[0] == ACTUAL], dtype=np.int8)
            grupos_kpartir.append((fut, pres))
        return self.sia_subsistema.kpartir(grupos_kpartir).distribucion_marginal()


# Alias canónico
KGeoMIP = GeometricSIAK