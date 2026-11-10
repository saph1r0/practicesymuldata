"""PASO 1 - Medir el problema ANTES de abrirlo.
Objetivo: saber si la data cabe en memoria y elegir la herramienta con numeros, no con intuicion."""
import os
import pandas as pd
import psutil
from comun import DATA, csv, conectar, cronometro

# 1) RAM de la maquina
ram = psutil.virtual_memory().total / 1e9
print(f"RAM total de esta maquina: {ram:.1f} GB\n")

# 2) Tamano en disco de cada archivo
print(f"{'archivo':<28}{'MB en disco':>12}")
for f in sorted(os.listdir(DATA)):
    if f.endswith(".csv"):
        print(f"{f:<28}{os.path.getsize(os.path.join(DATA, f)) / 1e6:>12,.0f}")

# 3) Cuantas filas tiene cada tabla (DuckDB cuenta sin cargar en memoria)
con = conectar()
with cronometro("conteo de filas"):
    filas = {}
    for t in ["maestro_empresas", "transacciones", "factoring_operaciones", "perfil_empresa"]:
        filas[t] = con.execute(f"SELECT count(*) FROM read_csv('{csv(t + '.csv')}', all_varchar=true)").fetchone()[0]
        print(f"  {t:<24} {filas[t]:>12,} filas")

# 4) Cuanto ocuparia cada tabla en pandas: leo 50 mil filas, mido y extrapolo
print(f"\n{'tabla':<24}{'columnas':>9}{'GB en pandas (estimado)':>26}")
total = 0
for t, n in filas.items():
    m = pd.read_csv(csv(t + ".csv"), nrows=50_000, low_memory=False)
    gb = m.memory_usage(deep=True).sum() / len(m) * n / 1e9
    total += gb
    print(f"{t:<24}{m.shape[1]:>9}{gb:>26.2f}")
print(f"{'TOTAL':<24}{'':>9}{total:>26.2f}")
print("\nRegla practica: pandas necesita 2 a 3 veces ese tamano para hacer joins y groupby.")
print(f"=> se necesitarian ~{2.5 * total:.0f} GB y la maquina tiene {ram:.0f} GB.")
