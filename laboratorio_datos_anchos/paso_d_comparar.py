"""D) Prueba final: el mismo LightGBM con distintas listas de columnas.
Se entrena con 2024 (80%), se valida con el 20% restante de 2024 y se mide en 2025 (lo que pasaria en produccion)."""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

prm = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=100, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=5, num_threads=2)
g = lambda yy, p: 2 * roc_auc_score(yy, p) - 1


def correr(nombre, df, feats):
    tr, te = df[df.anio_cohorte == 2024], df[df.anio_cohorte == 2025]
    idx = np.random.default_rng(0).random(len(tr)) < 0.8
    a, b = tr[idx], tr[~idx]
    t = time.time()
    m = lgb.train(prm, lgb.Dataset(a[feats], a.target), 3000, valid_sets=[lgb.Dataset(b[feats], b.target)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    seg = time.time() - t
    print(f"| {nombre} | {len(feats)} | {g(b.target, m.predict(b[feats])):.3f} | {g(te.target, m.predict(te[feats])):.3f} | {seg:.0f} s | {m.best_iteration} |", flush=True)


print("| Columnas usadas | N | Gini validación 2024 | Gini real 2025 | Tiempo | Árboles |")
crudo = pd.read_parquet("lab/clientes_ancho.parquet")
for c in crudo.columns:
    if crudo[c].dtype == object or str(crudo[c].dtype).startswith("str"):
        crudo[c] = crudo[c].astype("category")
correr("A. Todas, sin limpiar", crudo, [c for c in crudo.columns if c not in ("target", "anio_cohorte")])
del crudo
limpio = pd.read_parquet("lab/limpio.parquet")
correr("B. Todas, limpias", limpio, [c for c in limpio.columns if c not in ("target", "anio_cohorte")])
sel = pd.read_csv("lab/seleccionadas.csv").iloc[:, 0].tolist()
correr("C. Seleccionadas por el embudo", limpio, sel)
correr("D. Seleccionadas, sin l03 (revisión humana)", limpio, [c for c in sel if c != "l03"])
