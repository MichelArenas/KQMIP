import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from src.controllers.manager import Manager
from src.controllers.strategies.geometric import GeometricSIA

# Configuración manual para el primer subsistema de la hoja "3 elementos"
# Supongamos que es ABC_{t+1}|ABC_t
estado_inicial = "100"
condicion = "111"        # sin condicionar
# Para ABC_{t+1}|ABC_t, se conservan todas las futuras (A,B,C) y todas las presentes.
# En la máscara de GeometricSIA, 0 = conservar, 1 = marginalizar.
alcance = "000"          # conservar las tres primeras (bits 0,1,2)
mecanismo = "000"        # conservar las tres presentes

# Cargar TPM correcta (N3A.csv)
SAMPLES_DIR = Path("geomip/data/samples")   
if not tpm_path.exists():
    # buscar en .samples
    SAMPLES_DIR = Path("geomip/data/samples")   
tpm = np.genfromtxt(tpm_path, delimiter=",")
print(f"TPM cargada: {tpm_path}")

gestor = Manager(estado_inicial=estado_inicial)
estrategia = GeometricSIA(gestor)
sol = estrategia.aplicar_estrategia(condicion, alcance, mecanismo, tpm)
print(f"Partición: {sol.particion}")
print(f"Pérdida: {sol.perdida}")