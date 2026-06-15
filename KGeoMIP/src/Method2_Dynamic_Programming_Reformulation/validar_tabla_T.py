"""
Validación de la tabla de costos T contra la Tabla 4.2 del PDF2.
Usa los tensores del ejemplo (3 variables) y verifica los valores t(i,j) desde estado 000.
"""

import numpy as np

# ------------------------------------------------------------
# Datos del ejemplo (tomados del PDF2, Figura 4.4 y cálculos)
# Para cada variable (A, B, C), probabilidad de que la variable futura sea 1? 
# En el PDF usan "probabilidad de que la variable tome el valor 0".
# Pero las diferencias usan |X[i]-X[j]|, así que solo importan los valores.
# Los valores son los que extrajimos del PDF.
# Orden de los estados: 0=000, 1=100, 2=010, 3=001, 4=110, 5=101, 6=011, 7=111
# (en binario little‑endian, como se usa en el código)
tensor_A = np.array([0, 0, 0, 1, 0, 1, 1, 1], dtype=float)
tensor_B = np.array([0, 0, 1, 0, 1, 0, 1, 1], dtype=float)
tensor_C = np.array([0, 1, 0, 0, 1, 1, 0, 1], dtype=float)

# Valores esperados de t(000, j) para cada variable (Tabla 4.2)
expected = {
    'A': {
        0b000: 0.0,
        0b100: 0.0,
        0b010: 0.0,
        0b001: 0.5,
        0b110: 0.0,
        0b101: 0.375,
        0b011: 0.375,
        0b111: 0.21875,
    },
    'B': {
        0b000: 0.0,
        0b100: 0.0,
        0b010: 0.5,
        0b001: 0.0,
        0b110: 0.375,
        0b101: 0.0,
        0b011: 0.375,
        0b111: 0.21875,
    },
    'C': {
        0b000: 0.0,
        0b100: 0.5,
        0b010: 0.0,
        0b001: 0.0,
        0b110: 0.375,
        0b101: 0.375,
        0b011: 0.0,
        0b111: 0.21875,
    }
}

# ------------------------------------------------------------
# Implementación de la función de costo (copia de tu código)
def hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()

def get_vecinos_camino_optimo(i: int, j: int, n: int):
    """Vecinos de i que están en un camino óptimo hacia j (distancia Hamming decreciente)"""
    vecinos = []
    for bit in range(n):
        vecino = i ^ (1 << bit)   # cambiar el bit 'bit'
        if hamming_distance(vecino, j) < hamming_distance(i, j):
            vecinos.append(vecino)
    return vecinos

def calcular_costo(i: int, j: int, tensor: np.ndarray, n: int, memo: dict) -> float:
    """Función recursiva con memoización, igual a tu _calculator_costo"""
    if i == j:
        return 0.0
    key = (i, j)
    if key in memo:
        return memo[key]
    d = hamming_distance(i, j)
    gamma = 2.0 ** (-d)
    diff = abs(tensor[i] - tensor[j])
    vecinos = get_vecinos_camino_optimo(i, j, n)
    suma = 0.0
    for v in vecinos:
        suma += calcular_costo(v, j, tensor, n, memo)
    resultado = gamma * (diff + suma)
    memo[key] = resultado
    return resultado

def construir_tabla_costos(tensor: np.ndarray, n: int) -> dict:
    """Calcula t(i,j) para todos los pares (i,j) para una variable dada"""
    M = 1 << n
    memo = {}
    tabla = {}
    for i in range(M):
        for j in range(M):
            tabla[(i, j)] = calcular_costo(i, j, tensor, n, memo)
    return tabla

# ------------------------------------------------------------
# Validación
# ------------------------------------------------------------
def test_cost_table():
    n = 3
    # Calcular tablas completas para A, B, C
    tabla_A = construir_tabla_costos(tensor_A, n)
    tabla_B = construir_tabla_costos(tensor_B, n)
    tabla_C = construir_tabla_costos(tensor_C, n)
    
    i0 = 0  # estado inicial 000
    print("Validación de t(000, j) contra PDF2 (Tabla 4.2)")
    print("=" * 60)
    all_match = True
    for j in range(1, 1 << n):
        print(f"\nj = {j:03b} (decimal {j})")
        for var, tensor, tabla in [('A', tensor_A, tabla_A), ('B', tensor_B, tabla_B), ('C', tensor_C, tabla_C)]:
            esperado = expected[var][j]
            calculado = tabla[(i0, j)]
            match = abs(calculado - esperado) < 1e-6
            print(f"  {var}: esperado={esperado:.6f}, calculado={calculado:.6f} -> {'✓' if match else '✗'}")
            if not match:
                all_match = False
    if all_match:
        print("\n✅ ¡Todos los valores coinciden! La tabla T es correcta.")
    else:
        print("\n❌ Hay discrepancias. Revisa la implementación de la función de costo.")

if __name__ == "__main__":
    test_cost_table()