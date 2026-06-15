import sys
import os
from pathlib import Path
import time
import pandas as pd
import numpy as np

# Configurar el path para que pueda importar módulos desde src/
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.strategies.q_nodes_k import ParticionadorQ
from src.controllers.manager import Manager

def generar_tpm_masivo(n_bits, out_path):
    print(f"[{time.strftime('%H:%M:%S')}] Iniciando generación de TPM para N={n_bits}...")
    total_rows = 2**n_bits
    chunk_size = 2**18  # 262,144 filas por chunk para evitar sobrecarga de RAM
    
    with open(out_path, 'w') as f:
        for i in range(0, total_rows, chunk_size):
            rows = min(chunk_size, total_rows - i)
            chunk_data = np.random.rand(rows, n_bits).astype(np.float32)
            np.savetxt(f, chunk_data, delimiter=',', fmt='%.4f')
            
            progreso = min(i + chunk_size, total_rows)
            porcentaje = (progreso / total_rows) * 100
            print(f"  Progreso: {progreso:,} / {total_rows:,} ({porcentaje:.1f}%)")
            
    print(f"[{time.strftime('%H:%M:%S')}] Finalizado {out_path.name}. Tamaño: {os.path.getsize(out_path) / (1024*1024):.2f} MB")

def main():
    excel_path = project_root.parent / "DatosPruebas2026-1.xlsx"
    samples_dir = project_root / "src" / ".samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    
    if not excel_path.exists():
        print(f"Advertencia: No se encontró {excel_path}.")
        print(f"Se creará un nuevo archivo en {excel_path}.")
        
    ns = [10, 15, 20, 22, 25]
    ks = [2, 3, 4, 5]
    
    resultados_acumulados = []
    
    for n in ns:
        csv_path = samples_dir / f"N{n}A.csv"
        if not csv_path.exists():
            generar_tpm_masivo(n, csv_path)
            
        print(f"\n[{time.strftime('%H:%M:%S')}] --- Procesando Red Masiva N={n} ---")
        
        estado = "1" + "0" * (n - 1)
        gestor = Manager(estado)
        
        # Override tpm_filename manually for safety
        gestor.tpm_filename = csv_path
        
        print("Cargando TPM en memoria (esto puede tardar y consumir RAM para N=25)...")
        try:
            # pd.read_csv to np.ndarray is faster
            tpm = pd.read_csv(csv_path, header=None, dtype=np.float32).values
            print(f"TPM cargado exitosamente. Forma: {tpm.shape}")
        except Exception as e:
            print(f"Error al cargar la TPM: {e}")
            continue
        
        print("Inicializando ParticionadorQ...")
        cond = alc = mec = "1" * n
        estrategia = ParticionadorQ(tpm)
        
        for k in ks:
            print(f"[{time.strftime('%H:%M:%S')}] Evaluando QNodes para k={k}...")
            try:
                # ParticionadorQ internally uses time, but we wrap it too
                start_time = time.time()
                sol = estrategia.aplicar_estrategia(estado, cond, alc, mec, k=k)
                tiempo_total = time.time() - start_time
                
                perdida = sol.perdida
                particion_str = sol.particion if hasattr(sol, 'particion') else "N/A"
                
                print(f"    --> k={k} completado en {tiempo_total:.1f}s | φ={perdida:.4f}")
                resultados_acumulados.append({
                    "N (Nodos)": n,
                    "k (Particiones)": k,
                    "Pérdida (φ)": perdida,
                    "Estructura Partición": particion_str,
                    "Tiempo (s)": round(tiempo_total, 2)
                })
            except Exception as e:
                print(f"    --> Error en k={k}: {e}")
                resultados_acumulados.append({
                    "N (Nodos)": n,
                    "k (Particiones)": k,
                    "Pérdida (φ)": "ERROR",
                    "Estructura Partición": str(e),
                    "Tiempo (s)": 0
                })
                
        # Limpiar memoria de la TPM antes de pasar al siguiente N para evitar OutOfMemory
        del tpm
        import gc
        gc.collect()
                
    if resultados_acumulados:
        df_new = pd.DataFrame(resultados_acumulados)
        print("\nGuardando resultados en Excel...")
        try:
            if excel_path.exists():
                with pd.ExcelWriter(excel_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                    df_new.to_excel(writer, sheet_name="QNodes_Resultados", index=False)
            else:
                with pd.ExcelWriter(excel_path, mode='w', engine='openpyxl') as writer:
                    df_new.to_excel(writer, sheet_name="QNodes_Resultados", index=False)
            print(f"Resultados guardados exitosamente en la hoja 'QNodes_Resultados' de {excel_path.name}")
        except Exception as e:
            print(f"Error al escribir en el Excel: {e}")
            backup_file = project_root.parent / "backup_qnodes_masivos.csv"
            df_new.to_csv(backup_file, index=False)
            print(f"Se ha guardado un backup en {backup_file}")
    else:
        print("No se generaron resultados para guardar.")

if __name__ == "__main__":
    main()
