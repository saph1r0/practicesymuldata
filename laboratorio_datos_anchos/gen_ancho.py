"""Genera una tabla ANCHA de laboratorio: 500 mil filas x 200 columnas, con la 'verdad' de cada columna guardada
en verdad.csv para comprobar si las tecnicas de seleccion la descubren.
Uso: python gen_ancho.py --filas 500000 --salida lab"""
import argparse
import os
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--filas", type=int, default=500_000)
ap.add_argument("--salida", default="lab")
ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()
rng = np.random.default_rng(a.seed)
N = a.filas
os.makedirs(a.salida, exist_ok=True)
col, verdad = {}, []


def add(nombre, valores, tipo):
    col[nombre] = valores
    verdad.append((nombre, tipo))


cohorte = np.where(np.arange(N) < int(0.7 * N), 2024, 2025)      # 2024 = train, 2025 = test (corte temporal)
nuevo = cohorte == 2025

# ------------------ senal latente
S = rng.normal(0, 1, (N, 15))
S[:, 14] = rng.lognormal(10, 1.2, N)                    # s15: monto con cola larga
cats_sig = [rng.integers(0, k, N) for k in (5, 8, 12, 20, 6)]
efecto_cat = [rng.normal(0, 0.5, k) for k in (5, 8, 12, 20, 6)]
coef = np.array([1.0, .8, .7, .6, .5, .4, .35, .3, .2, .1])
logit = -4.3 + S[:, :10] @ coef
logit += 0.6 * (S[:, 10] ** 2 - 1)                      # s11: efecto en U (no lineal)
logit += 0.7 * S[:, 11] * S[:, 12]                      # s12 x s13: interaccion pura
logit += 0.9 * (S[:, 13] > 1.0)                         # s14: umbral
logit += 0.35 * (np.log(S[:, 14]) - 10)                 # s15: efecto en log
for c, e in zip(cats_sig, efecto_cat):
    logit += e[c]
z_presente = rng.random(N) < 0.04 + 0.04 * (logit > -2)   # la AUSENCIA de z01..z03 informa
logit += 0.0
e_lat = rng.normal(0, 1, N)
logit += 0.3 * e_lat                                     # senal que solo vive en columnas con centinelas
t_antig = rng.uniform(0, 20, N)
logit += -0.05 * (t_antig - 10)                          # antiguedad (vive en una fecha)
m_lat = rng.lognormal(8, 1, N)
logit += 0.25 * (np.log(m_lat) - 8)                      # monto escrito como texto con comas
y = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)

add("id_cliente", np.char.add("C", np.char.zfill((rng.permutation(N) + 10**8).astype(str), 10)), "id")
add("anio_cohorte", cohorte, "particion")
add("target", y, "target")
for j in range(15):
    add(f"s{j + 1:02d}", np.round(S[:, j], 4), "senal")
for j, c in enumerate(cats_sig):
    add(f"c{j + 1:02d}", np.char.add(f"G{j + 1}_", c.astype(str)), "senal_cat")
# redundantes: 3 versiones de s01..s10
for j in range(10):
    add(f"r{3 * j + 1:02d}", np.round(S[:, j] * 1000 + 50000, 1), "redundante")            # otra escala
    add(f"r{3 * j + 2:02d}", np.round(S[:, j] + rng.normal(0, 0.15, N), 4), "redundante")  # con ruido
    add(f"r{3 * j + 3:02d}", np.round(S[:, j] * 0.9 + S[:, (j + 1) % 10] * 0.1, 4), "redundante")
for j, src in enumerate(["s02", "s05", "c01", "s01", "c03"]):
    add(f"d{j + 1:02d}", col[src].copy(), "duplicada")
# ruido numerico y categorico
for j in range(39):
    v = [rng.normal(0, 1, N), rng.lognormal(3, 1, N), rng.integers(0, 100, N)][j % 3]
    add(f"n{j + 1:02d}", np.round(v, 3), "ruido")
for j in range(30):
    k = [3, 10, 50, 500, 5000][j % 5]
    add(f"k{j + 1:02d}", np.char.add("V", rng.integers(0, k, N).astype(str)), "ruido_cat")
# constantes y casi constantes
for j in range(10):
    v = np.full(N, "A", dtype=object) if j < 5 else np.where(rng.random(N) < 0.999, "X", "Y")
    add(f"q{j + 1:02d}", v, "constante")
# casi todo nulo: z01..z03 su presencia informa
for j in range(15):
    pres = z_presente if j < 3 else rng.random(N) < 0.03
    add(f"z{j + 1:02d}", np.where(pres, np.round(rng.normal(0, 1, N), 3), np.nan), "nula_informativa" if j < 3 else "nula")
# tipo identificador
add("i01", np.array([f"{x:016x}" for x in rng.integers(0, 2**62, N)]), "id")
add("i02", np.arange(N) + 5_000_000, "id")                                  # correlativo: sigue el orden de alta
add("i03", np.char.add("DOC-", np.arange(N).astype(str)), "id")
add("i04", rng.permutation(N), "id")
add("i05", np.char.add("TEL9", rng.integers(10**7, 10**8, N).astype(str)), "id")
# FUGA: informacion posterior al evento
# en 2025 (momento de predecir) estos campos aun no se llenan: asi pasa en la vida real
add("l01", np.where(nuevo, 0, np.where(y == 1, rng.choice([1, 0], N, p=[.9, .1]), rng.choice([1, 0], N, p=[.02, .98]))), "fuga")
add("l02", np.where((y == 1) & ~nuevo, rng.integers(1, 90, N), np.nan), "fuga")
add("l03", np.round(y * 0.8 * ~nuevo + rng.normal(0, 1, N), 3), "fuga")
# centinelas: e01..e03 con senal
for j in range(10):
    v = np.round((e_lat if j < 3 else rng.normal(0, 1, N)) * 100 + 500 + rng.normal(0, 20, N), 1).astype(object)
    r = rng.random(N)
    v[r < 0.08] = -999
    v[(r >= 0.08) & (r < 0.09)] = 99999999
    add(f"e{j + 1:02d}", v, "senal_centinela" if j < 3 else "ruido_centinela")
# fechas en formatos mezclados: t01 = antiguedad
base = np.datetime64("2025-06-30")
for j in range(10):
    dias = (t_antig * 365).astype(int) if j == 0 else rng.integers(0, 7000, N)
    f = pd.to_datetime(base - dias.astype("timedelta64[D]"))
    s = f.strftime("%Y-%m-%d").to_numpy(dtype=object)
    r = rng.random(N)
    s[r < 0.15] = f[r < 0.15].strftime("%d/%m/%Y")
    s[(r >= .15) & (r < .2)] = f[(r >= .15) & (r < .2)].strftime("%Y%m%d")
    s[(r >= .2) & (r < .21)] = "1900-01-01"
    add(f"t{j + 1:02d}", s, "senal_fecha" if j == 0 else "ruido_fecha")
# montos como texto con separador de miles: m01 = senal
for j in range(5):
    v = m_lat if j == 0 else rng.lognormal(8, 1, N)
    add(f"m{j + 1:02d}", np.array([f"{x:,.2f}" for x in v]), "senal_texto" if j == 0 else "ruido_texto")
# deriva: ruido cuya distribucion cambia en 2025
for j in range(15):
    add(f"w{j + 1:02d}", np.round(rng.normal(0, 1, N) + nuevo * (0.6 + 0.1 * j), 3), "deriva")

df = pd.DataFrame(col)
assert df.shape[1] == 200, df.shape
df.to_csv(os.path.join(a.salida, "clientes_ancho.csv"), index=False)
pd.DataFrame(verdad, columns=["columna", "tipo_real"]).to_csv(os.path.join(a.salida, "verdad.csv"), index=False)
print(df.shape, f"tasa target {y.mean():.2%}", f"MB {os.path.getsize(os.path.join(a.salida, 'clientes_ancho.csv')) / 1e6:,.0f}")
