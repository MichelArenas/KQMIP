# Parche dinámico: reemplazar QNodes por QNodesK en el módulo original
from src.strategies import q_nodes as qn_module
from src.strategies.q_nodes_k import QNodesK

qn_module.QNodes = QNodesK

# Luego ejecutar la aplicación normal
from src.models.base.application import aplicacion
from src.main import iniciar

def main():
    aplicacion.profiler_habilitado = True
    iniciar()

if __name__ == "__main__":
    main()