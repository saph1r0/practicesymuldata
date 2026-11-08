"""A) PERFIL AUTOMATICO de 200 columnas y CLASIFICACION de cada una.
Idea: nadie revisa 200 columnas a mano. Se calcula una ficha por columna y reglas la clasifican;
el humano solo revisa los casos dudosos."""
import time
import duckdb
import numpy as np
import pandas as pd

PQ = "lab/clientes_ancho.parquet"
t0 = time.time()
con = duckdb.connect()

# 1) Ficha basica de TODAS las columnas en una pasada (sobre las 500 mil filas)
s = con.execute(f"SUMMARIZE SELECT * FROM '{PQ}'").df()
ficha = s[["column_name", "column_type", "approx_unique", "null_percentage", "min", "max"]].rename(
    columns={"column_name": "col", "column_type": "tipo", "approx_unique": "distintos", "null_percentage": "pct_nulos"})
ficha["pct_nulos"] = ficha.pct_nulos.astype(float)
N = con.execute(f"SELECT count(*) FROM '{PQ}'").fetchone()[0]

# 2) Lo que SUMMARIZE no da, calculado sobre una MUESTRA de 100 mil filas (suficiente para proporciones)
m = con.execute(f"SELECT * FROM '{PQ}' USING SAMPLE 100000 ROWS (reservoir, 7)").df()
extra = []
for c in m.columns:
    x = m[c]
    vc = x.value_counts(dropna=True, normalize=True)
    fila = {"col": c, "pct_valor_top": round(100 * vc.iloc[0], 2) if len(vc) else 100.0,
            "pct_centinela": 0.0, "pct_parsea_numero": np.nan, "pct_parsea_fecha": np.nan}
    if x.dtype.kind in "if":
        fila["pct_centinela"] = round(100 * x.isin([-999, -9999, -1, 99999999, 999999999]).mean(), 2)
    else:
        txt = x.dropna().astype(str)
        fila["pct_parsea_numero"] = round(100 * pd.to_numeric(txt.str.replace(",", ""), errors="coerce").notna().mean(), 1)
        fila["pct_parsea_fecha"] = round(100 * pd.to_datetime(txt, errors="coerce", format="mixed", dayfirst=True).notna().mean(), 1) \
            if txt.str.contains(r"\d{4}|/").mean() > 0.9 else 0.0
    extra.append(fila)
ficha = ficha.merge(pd.DataFrame(extra), on="col")
ficha["ratio_unicos"] = ficha.distintos / N


# 3) Reglas de clasificacion (el orden importa: la primera que se cumple gana)
def clasificar(r):
    if r.col in ("target", "anio_cohorte"):
        return "control"
    if r.distintos <= 1 or r.pct_valor_top >= 99.5:
        return "constante"
    if r.pct_nulos >= 95:
        return "casi_nula"
    if r.ratio_unicos >= 0.95 and r.tipo in ("VARCHAR", "BIGINT"):
        return "identificador"
    if r.tipo == "VARCHAR" and r.pct_parsea_fecha >= 90:
        return "fecha_como_texto"
    if r.tipo == "VARCHAR" and r.pct_parsea_numero >= 95:
        return "numero_como_texto"
    if r.tipo == "VARCHAR":
        return "categorica_alta" if r.distintos > 50 else "categorica"
    if r.pct_centinela >= 0.5:
        return "numerica_con_centinela"
    return "numerica"


ficha["clase"] = ficha.apply(clasificar, axis=1)
ficha.to_csv("lab/ficha_columnas.csv", index=False)
print(f"Perfil de {len(ficha)} columnas en {time.time() - t0:.1f} s\n")
print(ficha.clase.value_counts().to_string())

verdad = pd.read_csv("lab/verdad.csv")
x = ficha.merge(verdad, left_on="col", right_on="columna")
print("\nComparacion con la verdad (filas = lo que la columna ES, columnas = como la clasificaron las reglas):")
print(pd.crosstab(x.tipo_real, x.clase).to_string())
print("\nEjemplos de ficha:")
print(ficha[ficha.col.isin(["s01", "e01", "m01", "t01", "z01", "q06", "i02", "k05", "c01"])][
    ["col", "tipo", "distintos", "pct_nulos", "pct_valor_top", "pct_centinela", "pct_parsea_numero", "pct_parsea_fecha", "clase"]].to_string(index=False))
