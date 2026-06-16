import sys
import os
os.environ["PYTHONIOENCODING"] = "utf-8"
from pathlib import Path
import multiprocessing
import numpy as np
import time
import traceback
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder='web', static_url_path='')

PROJECT_ROOT = Path(__file__).resolve().parent
QNODES_ROOT = PROJECT_ROOT / "KQNodes"
METHOD2_ROOT = PROJECT_ROOT / "KGeoMIP" / "src" / "Method2_Dynamic_Programming_Reformulation"

def worker_qnodes(estado_ini, condicion, alcance, mecanismo, tpm, k, queue):
    # Setup path
    sys.path.insert(0, str(QNODES_ROOT / "src"))
    sys.path.insert(0, str(QNODES_ROOT))
    try:
        from src.strategies.q_nodes_k import ParticionadorQ
        from src.models.base.application import aplicacion
        from src.models.core.solution import Solution
        Solution._Solution__anunciar_solucion = lambda self: None

        estrategia = ParticionadorQ(tpm)
        res = estrategia.aplicar_estrategia(estado_ini, condicion, alcance, mecanismo, k=k)
        
        queue.put({
            "estrategia": "ParticionQ",
            "k": k,
            "perdida": float(res.perdida),
            "particion": str(res.particion),
            "tiempo": float(res.tiempo_ejecucion),
            "distribucion_subsistema": res.distribucion_subsistema.tolist(),
            "distribucion_particion": res.distribucion_particion.tolist(),
            "error": None
        })
    except Exception as e:
        queue.put({"error": str(e), "traceback": traceback.format_exc()})

def worker_geometric(estado_ini, condicion, alcance, mecanismo, tpm, k, queue):
    # Setup path
    sys.path.insert(0, str(METHOD2_ROOT / "src"))
    sys.path.insert(0, str(METHOD2_ROOT))
    try:
        from src.controllers.strategies.geometric_k import GeometricSIAK
        from src.controllers.strategies.geometric import GeometricSIA
        from src.controllers.manager import Manager
        from src.models.core.solution import Solution
        Solution._Solution__anunciar_solucion = lambda self: None

        gestor = Manager(estado_inicial=estado_ini)
        # Patching tpm so it doesn't load it again
        # The manager doesn't store tpm, but SIA might load it.
        # GeometricSIAK(gestor).aplicar_estrategia receives tpm as argument.
        
        if k == 2:
            estrategia = GeometricSIA(gestor)
            res = estrategia.aplicar_estrategia(condicion, alcance, mecanismo, tpm)
        else:
            estrategia = GeometricSIAK(gestor)
            res = estrategia.aplicar_estrategia(condicion, alcance, mecanismo, tpm, k=k)
            
        queue.put({
            "estrategia": "KGeoMIP" if k > 2 else "GeometricSIA",
            "k": k,
            "perdida": float(res.perdida),
            "particion": str(res.particion),
            "tiempo": float(res.tiempo_ejecucion),
            "distribucion_subsistema": res.distribucion_subsistema.tolist(),
            "distribucion_particion": res.distribucion_particion.tolist(),
            "error": None
        })
    except Exception as e:
        queue.put({"error": str(e), "traceback": traceback.format_exc()})

def run_in_process(target, args, timeout=3600):
    queue = multiprocessing.Queue()
    proc = multiprocessing.Process(target=target, args=args + (queue,))
    proc.start()
    proc.join(timeout=timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join()
        return {"error": "Timeout"}
    if not queue.empty():
        return queue.get()
    return {"error": "Unknown error, empty queue"}

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/redes')
def get_redes():
    samples_dir = QNODES_ROOT / "src" / ".samples"
    redes = {}
    if samples_dir.exists():
        for file in samples_dir.glob("N*.csv"):
            name = file.stem
            if name.startswith('N'):
                import re
                m = re.match(r'N(\d+)([A-Z]+)', name)
                if m:
                    n_nodos = m.group(1)
                    pag = m.group(2)
                    if n_nodos not in redes:
                        redes[n_nodos] = []
                    redes[n_nodos].append(pag)
    for n in redes:
        redes[n].sort()
    return jsonify(redes)

@app.route('/api/ejecutar', methods=['POST'])
def ejecutar():
    data = request.json
    n_nodos = int(data.get('n_nodos', 3))
    estado_inicial = data.get('estado_inicial')
    condicion = data.get('condicion')
    alcance = data.get('alcance')
    mecanismo = data.get('mecanismo')
    k = int(data.get('k', 2))
    pagina = data.get('pagina', 'A')

    tpm_path = QNODES_ROOT / "src" / ".samples" / f"N{n_nodos}{pagina}.csv"
    if not tpm_path.exists():
        return jsonify({"error": f"No se encontro la red {tpm_path.name}"}), 404

    try:
        tpm = np.genfromtxt(str(tpm_path), delimiter=",")
    except Exception as e:
        return jsonify({"error": f"Error al cargar TPM: {e}"}), 500

    qnodes_res = run_in_process(worker_qnodes, (estado_inicial, condicion, alcance, mecanismo, tpm, k))
    geomip_res = run_in_process(worker_geometric, (estado_inicial, condicion, alcance, mecanismo, tpm, k))

    return jsonify({
        "qnodes": qnodes_res,
        "geometric": geomip_res
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
