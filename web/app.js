let redesDisponibles = {};

document.addEventListener('DOMContentLoaded', async () => {
    // Cargar redes disponibles
    try {
        const res = await fetch('/api/redes');
        redesDisponibles = await res.json();
        
        const nSelect = document.getElementById('n_nodos');
        Object.keys(redesDisponibles).sort((a,b) => parseInt(a)-parseInt(b)).forEach(n => {
            const opt = document.createElement('option');
            opt.value = n;
            opt.textContent = `N = ${n}`;
            nSelect.appendChild(opt);
        });
        
        if (nSelect.options.length > 0) {
            actualizarPaginas();
            autoCompletar();
        }

        nSelect.addEventListener('change', () => {
            actualizarPaginas();
            autoCompletar();
        });

    } catch (e) {
        console.error("Error cargando redes:", e);
    }

    document.getElementById('mip-form').addEventListener('submit', async (e) => {
        e.preventDefault();
        await ejecutarAnalisis();
    });
});

function actualizarPaginas() {
    const n = document.getElementById('n_nodos').value;
    const pSelect = document.getElementById('pagina');
    pSelect.innerHTML = '';
    
    if (redesDisponibles[n]) {
        redesDisponibles[n].forEach(p => {
            const opt = document.createElement('option');
            opt.value = p;
            opt.textContent = `Página ${p}`;
            pSelect.appendChild(opt);
        });
    }
}

function autoCompletar() {
    const n = parseInt(document.getElementById('n_nodos').value);
    if (isNaN(n)) return;

    // Estado inicial: 1 seguido de 0s
    document.getElementById('estado_inicial').value = "1" + "0".repeat(n - 1);
    // Cond, alc, mec: todos 1s
    const unos = "1".repeat(n);
    document.getElementById('condicion').value = unos;
    document.getElementById('alcance').value = unos;
    document.getElementById('mecanismo').value = unos;
}

async function ejecutarAnalisis() {
    const btn = document.getElementById('btn-ejecutar');
    const spinner = document.getElementById('spinner');
    const btnText = btn.querySelector('.btn-text');
    const resultsContainer = document.getElementById('results-container');
    
    // UI Loading state
    btn.disabled = true;
    spinner.classList.remove('hidden');
    btnText.textContent = "Analizando...";
    resultsContainer.style.display = 'none';

    // Reset winner classes
    document.getElementById('qnodes-result').classList.remove('winner');
    document.getElementById('geomip-result').classList.remove('winner');

    const data = {
        n_nodos: parseInt(document.getElementById('n_nodos').value),
        pagina: document.getElementById('pagina').value,
        k: parseInt(document.getElementById('k').value),
        estado_inicial: document.getElementById('estado_inicial').value,
        condicion: document.getElementById('condicion').value,
        alcance: document.getElementById('alcance').value,
        mecanismo: document.getElementById('mecanismo').value,
    };

    try {
        const res = await fetch('/api/ejecutar', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await res.json();
        
        if (result.error) {
            alert("Error del servidor: " + result.error);
            return;
        }

        renderResult('qn', result.qnodes);
        renderResult('geo', result.geometric);

        // Determinar ganador (menor phi > 0 o simplemente menor phi)
        // Tratamos null como infinito
        const phiQ = (result.qnodes.error || result.qnodes.perdida === null) ? Infinity : result.qnodes.perdida;
        const phiG = (result.geometric.error || result.geometric.perdida === null) ? Infinity : result.geometric.perdida;

        if (phiQ !== Infinity || phiG !== Infinity) {
            if (phiQ < phiG) {
                document.getElementById('qnodes-result').classList.add('winner');
            } else if (phiG < phiQ) {
                document.getElementById('geomip-result').classList.add('winner');
            } else {
                // Empate
                document.getElementById('qnodes-result').classList.add('winner');
                document.getElementById('geomip-result').classList.add('winner');
            }
        }

        resultsContainer.style.display = 'grid';

    } catch (e) {
        console.error(e);
        alert("Error de conexión al servidor");
    } finally {
        btn.disabled = false;
        spinner.classList.add('hidden');
        btnText.textContent = "Ejecutar Análisis";
    }
}

function renderResult(prefix, data) {
    const errorEl = document.getElementById(`${prefix}-error`);
    const bodyEl = document.querySelector(`#${prefix === 'qn' ? 'qnodes' : 'geomip'}-result .card-body`);
    
    if (data.error) {
        errorEl.style.display = 'block';
        errorEl.textContent = data.error;
        bodyEl.style.display = 'none';
        return;
    }

    errorEl.style.display = 'none';
    bodyEl.style.display = 'block';

    document.getElementById(`${prefix}-strat`).textContent = data.estrategia;
    document.getElementById(`${prefix}-phi`).textContent = data.perdida !== null ? data.perdida.toFixed(6) : "N/A";
    document.getElementById(`${prefix}-time`).textContent = data.tiempo !== null ? data.tiempo.toFixed(4) : "N/A";
    document.getElementById(`${prefix}-part`).textContent = data.particion || "N/A";
    
    const formatArr = (arr) => {
        if (!arr) return "N/A";
        return "[ " + arr.map(v => v.toFixed(4)).join(", ") + " ]";
    };

    document.getElementById(`${prefix}-dist-sub`).textContent = formatArr(data.distribucion_subsistema);
    document.getElementById(`${prefix}-dist-part`).textContent = formatArr(data.distribucion_particion);
}
