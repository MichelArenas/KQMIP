# exec_k.py
# Este archivo parchea la estrategia geométrica original para usar la versión extendida (k-particiones)
# Sin modificar archivos originales. Ejecútalo con: .\.venv\Scripts\python.exe exec_k.py



# 1. Parche: reemplazar la clase GeometricSIA en el módulo original
from src.controllers.strategies import geometric as geometric_module
from src.controllers.strategies.geometric_k import GeometricSIAK

# Reemplazo en caliente
geometric_module.GeometricSIA = GeometricSIAK

# 2. Ahora importamos el resto de la aplicación normalmente
from src.models.base.application import aplicacion
from src.main import iniciar

def main():
    aplicacion.profiler_habilitado = True
    # Si quieres forzar modo manual o algo, puedes setear variables aquí
    iniciar()

if __name__ == "__main__":
    main()