"""PASO 2 - Perfilar cada tabla sin cargarla.
Objetivo: por cada columna saber tipo real, % de nulos, cuantos valores distintos y extremos.
Eso dice que columnas hay que limpiar y como."""
import sys
import pandas as pd
from comun import csv, conectar, cronometro

pd.set_option("display.width", 200, "display.max_columns", 20, "display.max_colwidth", 30)
con = conectar()
tablas = sys.argv[1:] or ["maestro_empresas", "transacciones", "factoring_operaciones", "perfil_empresa"]

for t in tablas:
    print("=" * 100, f"\n{t}\n" + "=" * 100)
    src = f"read_csv('{csv(t + '.csv')}', all_varchar=true)"
    with cronometro(f"SUMMARIZE {t}"):
        # SUMMARIZE: min, max, distintos aproximados y % de nulos de TODAS las columnas en una pasada
        s = con.execute(f"SUMMARIZE SELECT * FROM {src}").df()
    s = s[["column_name", "min", "max", "approx_unique", "null_percentage"]]
    if t == "perfil_empresa":            # 116 columnas: mostramos solo las no mensuales + un mes
        s = s[~s.column_name.str.contains(r"_m(0[2-9]|1[0-2])$")]
    print(s.to_string(index=False))
    # columnas con pocos valores distintos: ver TODOS sus valores (aqui aparecen las variantes sucias)
    pocas = s[(s.approx_unique <= 12) & (s.approx_unique > 1)].column_name.tolist()
    for c in pocas[:8]:
        vc = con.execute(f"SELECT {c} AS v, count(*) n FROM {src} GROUP BY 1 ORDER BY 2 DESC").df()
        print(f"\n  valores de {c}: " + ", ".join(f"{v!r}={n:,}" for v, n in zip(vc.v, vc.n)))
