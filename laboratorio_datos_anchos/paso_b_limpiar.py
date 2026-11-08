"""B) LIMPIEZA POR REGLAS: cada CLASE de columna (del perfil) tiene UNA funcion de limpieza.
200 columnas no se limpian una por una: se limpian 8 clases."""
import time
import resource
import numpy as np
import pandas as pd

t0 = time.time()
ficha = pd.read_csv("lab/ficha_columnas.csv")
clase = dict(zip(ficha.col, ficha.clase))
descartar = [c for c, k in clase.items() if k in ("constante", "identificador")]
leer = [c for c in ficha.col if c not in descartar]
# column pruning: Parquet permite leer SOLO las columnas que sirven
df = pd.read_parquet("lab/clientes_ancho.parquet", columns=leer)
print(f"descartadas al leer (constantes + identificadores): {len(descartar)} -> se leen {len(leer)} columnas")
CORTE = pd.Timestamp("2025-06-30")
log = []
nuevas, quitar = {}, []          # columnas nuevas se juntan y se agregan UNA vez (evita fragmentar)
for c in leer:
    k = clase[c]
    if k == "numero_como_texto":
        df[c] = pd.to_numeric(df[c].str.replace(",", "", regex=False), errors="coerce").astype("float32")
    elif k == "fecha_como_texto":
        s = df[c]
        f = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")             # formatos explicitos: ~10x mas
        f = f.fillna(pd.to_datetime(s, format="%d/%m/%Y", errors="coerce"))   # rapido que format="mixed"
        f = f.fillna(pd.to_datetime(s, format="%Y%m%d", errors="coerce"))
        f = f.where(f.dt.year > 1900)                                          # centinela de fecha
        nuevas[c + "_dias"] = (CORTE - f).dt.days.astype("float32")            # fecha -> numero util
        quitar.append(c)
        log.append((c, f"fecha -> {c}_dias; no parseadas {f.isna().mean():.1%}"))
    elif k == "numerica_con_centinela":
        n = df[c].isin([-999, -9999, 99999999, 999999999]).sum()
        df[c] = df[c].where(~df[c].isin([-999, -9999, 99999999, 999999999])).astype("float32")
        log.append((c, f"{n:,} centinelas -> nulo"))
    elif k == "casi_nula":
        nuevas[c + "_presente"] = df[c].notna().astype("int8")                # la AUSENCIA puede informar
        quitar.append(c)
    elif k == "categorica_alta":
        # >50 categorias: los arboles memorizan categorias raras. Se reemplaza por su FRECUENCIA (cuantas filas la tienen)
        nuevas[c + "_frec"] = df[c].map(df[c].value_counts()).astype("float32")
        quitar.append(c)
    elif k == "categorica":
        df[c] = df[c].astype("category")
    elif k == "numerica":
        if df[c].dtype == "float64":
            df[c] = df[c].astype("float32")
df = pd.concat([df.drop(columns=quitar), pd.DataFrame(nuevas)], axis=1)
for c, msg in log[:3] + log[10:12]:
    print(f"  {c}: {msg}")
df.to_parquet("lab/limpio.parquet", index=False)
print(f"tabla limpia: {df.shape}, {df.memory_usage(deep=True).sum() / 1e9:.2f} GB en memoria | "
      f"{time.time() - t0:.1f} s | pico {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")
