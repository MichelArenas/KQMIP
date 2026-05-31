import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Ajusta la ruta de importación según tu proyecto
sys.path.insert(0, os.path.abspath('.'))

from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA   # original, para k=2

# ------------------------------------------------------------
# Configuración
# ------------------------------------------------------------
EXCEL_PATH = Path("DatosPruebas2026_1.xlsx")        # Ruta de la plantilla
SAMPLES_DIR = Path("geomip/data/samples")           # Carpeta con los CSVs (ajústala)

def get_csv_path(sheet_name: str) -> Path:
    """
    Mapea el nombre de la hoja al archivo CSV correspondiente.
    Ejemplos:
        "10A-Elementos" -> N10A.csv
        "15B-Elementos" -> N15B.csv
        "20A-Elementos" -> N20A.csv
        "22A-Elementos" -> N22A.csv
        "25A-Elementos" -> N25A.csv
    """
    # Extraer la parte antes del guion, ej. "10A"
    base = sheet_name.split('-')[0]
    # Si la hoja se llama "15B-Elementos", base = "15B"
    csv_name = f"N{base}.csv"
    return SAMPLES_DIR / csv_name

def construir_bitmask(indices_conservar: list, total: int) -> str:
    """Construye cadena de bits donde '0' = conservar, '1' = marginalizar."""
    bits = ['1'] * total
    for i in indices_conservar:
        if 0 <= i < total:
            bits[i] = '0'
    return ''.join(bits)

def parse_subsistema(cadena: str, total_vars: int):
    """
    Parsea cadenas como "ABCDEFGHIJ" o "ABCDEFGHIJ|ABDEGHJ".
    Devuelve (futuros, presentes) como listas de índices.
    """
    if '|' in cadena:
        fut_str, pres_str = cadena.split('|')
    else:
        fut_str = pres_str = cadena
    fut = [ord(ch) - ord('A') for ch in fut_str if ch.isalpha()]
    pres = [ord(ch) - ord('A') for ch in pres_str if ch.isalpha()]
    return fut, pres

# ------------------------------------------------------------
# Procesamiento principal
# ------------------------------------------------------------
def main():
    # Verificar que exista la plantilla
    if not EXCEL_PATH.exists():
        print(f"Error: No se encuentra {EXCEL_PATH}")
        return

    # Leer todas las hojas
    xls = pd.ExcelFile(EXCEL_PATH)
    # Hojas objetivo (las que tienen "Elementos" y no son "Requerimientos" ni "plataformas")
    hojas = [sh for sh in xls.sheet_names if 'Elementos' in sh and not sh.startswith('plataformas')]
    print(f"Hojas a procesar: {hojas}")

    for hoja in hojas:
        print(f"\n=== Procesando hoja: {hoja} ===")
        # Leer la hoja sin cabeceras (para acceder a filas de metadatos)
        df = pd.read_excel(xls, sheet_name=hoja, header=None)

        # Extraer metadatos: estado inicial en celda B2, sistema en B3
        estado_inicial = df.iloc[1, 1]   # B2
        sistema_nombre = df.iloc[2, 1]   # B3 (ej. "ABCDEFGHIJ")
        num_vars = len(sistema_nombre)

        # Ajustar estado inicial si no tiene la longitud correcta
        if isinstance(estado_inicial, str) and len(estado_inicial) != num_vars:
            estado_inicial = "1" + "0"*(num_vars-1)
            print(f"  Estado inicial ajustado a: {estado_inicial}")

        # Determinar el archivo CSV a partir del nombre de la hoja
        csv_path = get_csv_path(hoja)
        if not csv_path.exists():
            print(f"  ¡Advertencia! No se encuentra {csv_path}. Se omite esta hoja.")
            continue
        print(f"  Usando CSV: {csv_path}")
        tpm = np.genfromtxt(csv_path, delimiter=",")

        # Crear gestor (se reutilizará para todas las pruebas de esta hoja)
        gestor = Manager(estado_inicial)

        # Localizar las filas de datos: comienzan en la fila 4 (índice 3) donde la columna B (Alcance) no es NaN
        datos_rows = []
        for idx in range(4, len(df)):
            alcance_str = df.iloc[idx, 1]   # columna B
            if pd.isna(alcance_str):
                break
            mecanismo_str = df.iloc[idx, 2] # columna C
            datos_rows.append((idx, alcance_str, mecanismo_str))

        print(f"  Se encontraron {len(datos_rows)} pruebas.")

        # Columnas para Geometric (k=2) según la plantilla: G(6), H(7), I(8)
        col_part = 6
        col_loss = 7
        col_time = 8

        # Instanciar la estrategia original (una sola vez)
        estrategia = GeometricSIA(gestor)

        # Procesar cada fila
        for row_idx, alcance_str, mecanismo_str in datos_rows:
            print(f"    Procesando fila {row_idx+1}...")
            # Parsear las cadenas del subsistema
            try:
                fut_keep, pres_keep = parse_subsistema(alcance_str, num_vars)
                # Si mecanismo_str está vacío o es NaN, usar el mismo alcance_str
                if pd.isna(mecanismo_str) or mecanismo_str == '':
                    mecanismo_str = alcance_str
                _, pres_keep = parse_subsistema(mecanismo_str, num_vars)
            except Exception as e:
                print(f"      Error parseando: {e}")
                continue

            # Construir las cadenas de bits para condicion, alcance, mecanismo
            condicion = '1' * num_vars        # sin condicionar
            alcance_bits = construir_bitmask(fut_keep, num_vars)
            mecanismo_bits = construir_bitmask(pres_keep, num_vars)

            try:
                inicio = time.perf_counter()
                # Nota: GeometricSIA original NO recibe k, solo cond, alc, mec, tpm
                sol = estrategia.aplicar_estrategia(condicion, alcance_bits, mecanismo_bits, tpm)
                elapsed = time.perf_counter() - inicio

                # Guardar resultados en el DataFrame
                df.iat[row_idx, col_part] = sol.particion
                df.iat[row_idx, col_loss] = sol.perdida
                df.iat[row_idx, col_time] = elapsed
                print(f"        Pérdida={sol.perdida:.6f}, tiempo={elapsed:.2f}s")
            except Exception as e:
                print(f"      Error ejecutando: {e}")
                continue

        # Guardar la hoja modificada (reescribiendo el archivo completo)
        with pd.ExcelWriter(EXCEL_PATH, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
            df.to_excel(writer, sheet_name=hoja, index=False, header=False)
        print(f"  Hoja {hoja} guardada.")

    print("\n¡Proceso completado!")

if __name__ == "__main__":
    main()