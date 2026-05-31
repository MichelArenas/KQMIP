import subprocess
import pandas as pd
import time
import os

# Configuración de rutas
RUTA_QNODES = "./QNodes"
RUTA_GEOMIP = "./GeoMIP/src/Method2_Dynamic_Programming_Reformulation"
EXCEL_GEOMIP_SALIDA = "./GeoMIP/results/resultados_Geometric.xlsx"

def ejecutar_qnodes(estado, cond, alc, mec):
    """Modifica temporalmente main.py de QNodes y lo ejecuta."""
    # Aquí puedes automatizar la edición de QNodes/src/main.py si lo deseas, 
    # o asegurarte de que procese el mismo caso de prueba.
    print(f"Ejecutando QNodes para Estado: {estado}...")
    start_time = time.time()
    
    # Ejecutar QNodes usando uv
    resultado = subprocess.run(
        ["uv", "run", "exec.py"], 
        cwd=RUTA_QNODES, 
        capture_output=True, 
        text=True
    )
    tiempo = time.time() - start_time
    
    # Parsear la consola de QNodes para extraer la Pérdida (Phi/EMD)
    # (Ajusta este parseo según cómo imprima tu exec.py)
    perdida = 0.0
    for linea in resultado.stdout.split("\n"):
        if "perdida" in linea.lower() or "phi" in linea.lower():
            try:
                perdida = float(linea.split("=")[-1].strip())
            except:
                pass
                
    return tiempo, perdida

def ejecutar_geomip():
    """Ejecuta el lote de GeoMIP que lee del Excel."""
    print("Ejecutando GeoMIP desde Excel...")
    resultado = subprocess.run(
        ["uv", "run", "exec.py"], 
        cwd=RUTA_GEOMIP, 
        capture_output=True, 
        text=True
    )
    # GeoMIP escribe directo en resultados_Geometric.xlsx, así que leemos ese archivo
    df_resultados = pd.read_excel(EXCEL_GEOMIP_SALIDA)
    return df_resultados

# --- FLUJO DE EJECUCIÓN DEL EXPERIMENTO ---
# 1. Corres GeoMIP para procesar la hoja de cálculo masiva
df_geo = ejecutar_geomip()

# 2. Iteras sobre los mismos casos en QNodes para comparar (Muestras de control)
# Nota: Como QNodes es fuerza bruta, hazlo solo para los tamaños pequeños (N2, N3, N4)
resultados_comparativos = []

# Supongamos que tu excel de GeoMIP tiene columnas: 'Tamaño', 'Estado_Inicial', 'K_Particion', 'Tiempo_Geo', 'Perdida_Geo'
for index, fila in df_geo.iterrows():
    if fila['Tamaño'] <= 4: # Filtrar para evitar congelar QNodes con N5 o superior
        t_qnodes, p_qnodes = ejecutar_qnodes(
            fila['Estado_Inicial'], 
            fila['Condicion'], 
            fila['Alcance'], 
            fila['Mecanismo']
        )
        
        resultados_comparativos.append({
            "Tamaño": fila['Tamaño'],
            "K": fila['K_Particion'],
            "Tiempo_QNodes": t_qnodes,
            "Perdida_QNodes": p_qnodes,
            "Tiempo_GeoMIP": fila['Tiempo_Geo'],
            "Perdida_GeoMIP": fila['Perdida_Geo']
        })

# Guardar matriz final para gráficos
df_final = pd.DataFrame(resultados_comparativos)
df_final.to_csv("matriz_comparativa_ada.csv", index=False)
print("¡Pruebas completadas! Datos guardados en matriz_comparativa_ada.csv")