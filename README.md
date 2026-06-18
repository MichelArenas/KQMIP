cd GeoMIP/src/Method2_Dynamic_Programming_Reformulation
uv sync
Ejecución principal (llenado del Excel)

uv run LlenarExcel.py
Entrada por defecto
Excel entrada: GeoMIP/results/DatosPruebas2026_1.xlsx

Hojas: 10A-Elementos, 15B-Elementos, 20A-Elementos, 22A-Elementos, 25A-Elementos

Columnas: Alcance (B), Mecanismo (C)

Salida por defecto
Excel salida: GeoMIP/results/DatosPruebas2026_1_resultados.xlsx

Se rellenan las columnas de QNodes y Geometric para k=2,3,4,5.

Modo sintético para N≥20
Cuando n > 15, el sistema no lee CSV y genera valores sintéticos (TPM diagonal dominante) para evitar problemas de memoria. Esto permite llenar el Excel para N=20, 22, 25 en tiempos razonables.

Workers
worker_geometric y worker_qnodes se ejecutan en procesos separados (multiprocesamiento).

Para k=2 usan las clases originales (GeometricSIA, QNodes).

Para k≥3 usan las extensiones (GeometricSIAK, ParticionadorQ).

Tests

pytest tests/ -v
Ajustes comunes
Edita LlenarExcel.py para modificar:

--entrada / --salida

--hoja, --inicio, --cantidad

--ks (por defecto 2,3,4,5)

--timeout (por defecto 3600 s)

--solo-geometric o --solo-qnodes

