"""Mide tiempo y memoria pico de distintas formas de leer la MISMA tabla de 200 columnas.
Cada forma corre en un proceso aparte para que la memoria pico de una no contamine a otra.
Uso: python exp_memoria.py            (corre todo y muestra la tabla)"""
import os
import sys
import time
import resource
import subprocess
import json

CSV = "lab/clientes_ancho.csv"
PQ = "lab/clientes_ancho.parquet"
COLS10 = ["target", "anio_cohorte", "s01", "s02", "s03", "c01", "c02", "e01", "m01", "t01"]


def medir(modo):
    import numpy as np
    import pandas as pd
    import duckdb
    t = time.time()
    info = ""
    if modo == "pandas_todo":
        df = pd.read_csv(CSV, low_memory=False)
        info = f"{df.shape}, {df.memory_usage(deep=True).sum() / 1e9:.2f} GB en el DataFrame; tipos: {df.dtypes.astype(str).value_counts().to_dict()}"
    elif modo == "pandas_10_columnas":
        df = pd.read_csv(CSV, usecols=COLS10)
        info = f"{df.shape}, {df.memory_usage(deep=True).sum() / 1e9:.3f} GB"
    elif modo == "pandas_tipos_optimizados":
        # 1) leer una muestra para decidir tipos  2) leer todo con esos tipos
        m = pd.read_csv(CSV, nrows=20_000, low_memory=False)
        dt = {}
        for c in m.columns:
            if m[c].dtype == object and m[c].nunique() < 1000:
                dt[c] = "category"                       # texto repetido -> codigos enteros
            elif m[c].dtype == "float64":
                dt[c] = "float32"                        # la mitad de bytes
            elif m[c].dtype == "int64":
                dt[c] = "int32"
        df = pd.read_csv(CSV, dtype=dt, low_memory=False)
        info = f"{df.shape}, {df.memory_usage(deep=True).sum() / 1e9:.2f} GB en el DataFrame"
    elif modo == "pandas_por_bloques":
        # agregacion sin cargar todo: tasa de target por c01, leyendo de a 50 mil filas
        acc = None
        for ch in pd.read_csv(CSV, usecols=["c01", "target"], chunksize=50_000):
            g = ch.groupby("c01").target.agg(["sum", "count"])
            acc = g if acc is None else acc.add(g, fill_value=0)
        info = f"tasa por c01 calculada con {len(acc)} grupos"
    elif modo == "duckdb_csv_a_parquet":
        con = duckdb.connect()
        con.execute(f"COPY (SELECT * FROM read_csv('{CSV}', all_varchar=false, sample_size=-1)) TO '{PQ}' (FORMAT parquet, COMPRESSION zstd)")
        info = f"Parquet de {os.path.getsize(PQ) / 1e6:.0f} MB (CSV: {os.path.getsize(CSV) / 1e6:.0f} MB)"
    elif modo == "parquet_todo":
        df = pd.read_parquet(PQ)
        info = f"{df.shape}"
    elif modo == "parquet_10_columnas":
        df = pd.read_parquet(PQ, columns=COLS10)
        info = f"{df.shape}"
    elif modo == "duckdb_agregar_csv":
        r = duckdb.sql(f"SELECT c01, avg(target) FROM read_csv('{CSV}') GROUP BY 1").fetchall()
        info = f"{len(r)} grupos"
    elif modo == "duckdb_agregar_parquet":
        r = duckdb.sql(f"SELECT c01, avg(target) FROM '{PQ}' GROUP BY 1").fetchall()
        info = f"{len(r)} grupos"
    elif modo == "duckdb_filtrar_parquet":
        r = duckdb.sql(f"SELECT count(*), avg(s01) FROM '{PQ}' WHERE anio_cohorte = 2025 AND s15 > 100000").fetchall()
        info = f"{r[0][0]:,} filas cumplen"
    seg = time.time() - t
    gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(json.dumps({"modo": modo, "seg": round(seg, 1), "gb_pico": round(gb, 2), "info": info}))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        medir(sys.argv[1])
    else:
        for modo in ["pandas_todo", "pandas_10_columnas", "pandas_tipos_optimizados", "pandas_por_bloques",
                     "duckdb_csv_a_parquet", "parquet_todo", "parquet_10_columnas",
                     "duckdb_agregar_csv", "duckdb_agregar_parquet", "duckdb_filtrar_parquet"]:
            out = subprocess.run([sys.executable, __file__, modo], capture_output=True, text=True)
            print(out.stdout.strip() or out.stderr.strip()[-300:])
