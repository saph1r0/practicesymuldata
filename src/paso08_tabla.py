"""PASO 8 - Tabla de modelado: una fila por empresa, todas las features + target.
Siempre LEFT JOIN desde el maestro: el universo es el maestro, y no tener transacciones ES informacion."""
import numpy as np
from comun import pq, FEAT, conectar, cronometro

con = conectar()
F = lambda n: f"'{pq(FEAT, n)}'"
with cronometro("join"):
    df = con.execute(f"""
      SELECT * FROM {F('f_maestro')}
      LEFT JOIN {F('f_red')}       USING (k)
      LEFT JOIN {F('f_pagador')}   USING (k)
      LEFT JOIN {F('f_transf')}    USING (k)
      LEFT JOIN {F('f_factoring')} USING (k)
      LEFT JOIN {F('f_perfil')}    USING (k)
      LEFT JOIN {F('target')}      USING (k)""").df()

# Nulo por "no hay eventos" = 0 (no tuvo facturas); nulo por "no se sabe" (score, saldo) se deja nulo
cero = ["n_facturas", "vol_facturas", "n_pagadores", "vol_estado", "vol_ult3m", "vol_ult6m", "vol_prev6m",
        "meses_activos", "vol_pagado_facturas", "n_proveedores", "n_transf_in", "n_transf_out",
        "n_ops_bcp", "n_ops_otro", "monto_bcp_hist", "monto_otro_hist", "n_vencidas"]
df[cero] = df[cero].fillna(0)
# Variables derivadas (ratios de negocio)
df["log_vol_facturas"] = np.log1p(df.vol_facturas)
df["crec_facturas"] = (df.vol_ult6m - df.vol_prev6m) / (df.vol_prev6m + 1)
df["facturas_sobre_ventas"] = df.vol_facturas / 1.5 / df.ventas          # 18 meses -> anual aprox
df["es_pagador"] = (df.vol_pagado_facturas > df.vol_facturas).astype("int8")
df["ya_factoriza_otro"] = (df.n_ops_otro > 0).astype("int8")

# --------- CONTROLES (si uno falla, se detiene todo: mejor fallar aqui que en el modelo)
assert df.k.is_unique, "llave duplicada tras los joins"
n_maestro = con.execute(f"SELECT count(*) FROM {F('f_maestro')}").fetchone()[0]
assert len(df) == n_maestro, f"se perdieron/ganaron filas: {len(df)} vs {n_maestro} del maestro"
assert df.target.notna().all(), "empresas sin target asignado"
assert df[df.es_test == 1].target.sum() == 0, "hay target en test: fuga"
nulos = df.isna().mean().sort_values(ascending=False)
print(f"tabla: {df.shape[0]:,} filas x {df.shape[1]} columnas | memoria {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
print("columnas con mas nulos:\n" + nulos.head(8).round(3).to_string())
const = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
print("columnas constantes (se descartan):", const)
df = df.drop(columns=const)
df.to_parquet(pq(FEAT, "tabla_modelo"), index=False)
print("-> features/tabla_modelo.parquet")
