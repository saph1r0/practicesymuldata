"""C) EMBUDO DE SELECCION: de ~180 columnas a las que importan.
Regla de oro: TODAS las decisiones se toman mirando SOLO train (cohorte 2024).
La cohorte 2025 se usa para dos cosas legitimas: (1) comparar DISTRIBUCIONES de las variables (no el target)
y (2) medir el resultado final."""
import time
import hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

T0 = time.time()
df = pd.read_parquet("lab/limpio.parquet")
verdad = pd.read_csv("lab/verdad.csv").set_index("columna").tipo_real
base = lambda c: c.replace("_dias", "").replace("_presente", "").replace("_frec", "")
tr, te = df[df.anio_cohorte == 2024].reset_index(drop=True), df[df.anio_cohorte == 2025].reset_index(drop=True)
y = tr.target.values
feats = [c for c in df.columns if c not in ("target", "anio_cohorte")]
embudo = [("inicio (tras limpieza)", len(feats), "")]


def gini_univariado(x):
    if str(x.dtype) == "category":                       # categorica: media del target por categoria, fuera de pliegue
        enc = np.zeros(len(x))
        for a, b in KFold(5, shuffle=True, random_state=1).split(x):
            medias = pd.Series(y[a]).groupby(x.iloc[a].values, observed=True).mean()
            enc[b] = x.iloc[b].map(medias).astype(float).fillna(y[a].mean()).values
        return abs(2 * roc_auc_score(y, enc) - 1)
    v = x.astype(float).values
    v = np.where(np.isnan(v), np.nanmin(v) - 1, v)      # el nulo cuenta como su propio valor
    return abs(2 * roc_auc_score(y, v) - 1)


# ---------------------------------------------------------------- F1 duplicadas exactas
t = time.time()
huellas, dup = {}, []
for c in feats:
    # huella que depende de CADA valor y su posicion (sumar hashes no sirve: dos 0/1 con igual cantidad de unos chocan)
    h = hashlib.md5(pd.util.hash_pandas_object(tr[c].astype(str) if str(tr[c].dtype) == "category" else tr[c], index=False).values.tobytes()).hexdigest()
    if h in huellas:
        dup.append(c)
    else:
        huellas[h] = c
feats = [c for c in feats if c not in dup]
embudo.append(("F1 duplicadas exactas", len(feats), ", ".join(dup)))
print(f"F1 duplicadas: {dup}  ({time.time() - t:.1f} s)")

# ---------------------------------------------------------------- Gini univariado (se usa en F2, F4 y F5)
t = time.time()
gu = pd.Series({c: gini_univariado(tr[c]) for c in feats}).sort_values(ascending=False)
print(f"\nGini univariado de {len(feats)} columnas en {time.time() - t:.1f} s. Top 12:")
print(pd.DataFrame({"gini": gu.head(12).round(3), "es_en_verdad": [verdad.get(base(c)) for c in gu.head(12).index]}).to_string())


# ---------------------------------------------------------------- F2 deriva / fuga por cambio de distribucion (PSI)
def psi(a, b):
    """Population Stability Index: compara la distribucion de una variable en dos periodos."""
    if str(a.dtype) == "category":
        pa = a.astype(str).value_counts(normalize=True)
        pb = b.astype(str).value_counts(normalize=True).reindex(pa.index).fillna(0)
    else:
        a, b = a.astype(float).values, b.astype(float).values
        cortes = np.unique(np.nanquantile(a, np.linspace(0, 1, 11))[1:-1])   # 10 tramos con igual % en train
        ia = np.where(np.isnan(a), len(cortes) + 1, np.searchsorted(cortes, a))    # el nulo es su propio tramo
        ib = np.where(np.isnan(b), len(cortes) + 1, np.searchsorted(cortes, b))
        pa = pd.Series(np.bincount(ia, minlength=len(cortes) + 2) / len(a))
        pb = pd.Series(np.bincount(ib, minlength=len(cortes) + 2) / len(b))
    pa, pb = pa.clip(lower=1e-4), pb.clip(lower=1e-4)
    return float(((pa - pb) * np.log(pa / pb)).sum())


t = time.time()
ps = pd.Series({c: psi(tr[c], te[c]) for c in feats}).sort_values(ascending=False)
inestables = ps[ps > 0.25].index.tolist()
print(f"\nF2 PSI 2024 vs 2025 ({time.time() - t:.1f} s). Mayores:")
print(pd.DataFrame({"psi": ps.head(8).round(3), "es_en_verdad": [verdad.get(base(c)) for c in ps.head(8).index]}).to_string())
feats = [c for c in feats if c not in inestables]
embudo.append(("F2 inestables (PSI > 0.25)", len(feats), ", ".join(inestables)))

# ---------------------------------------------------------------- F3 fuga por Gini sospechoso (las que quedan)
sospechosas = [c for c in gu.index if c in feats and gu[c] > 0.5]
feats = [c for c in feats if c not in sospechosas]
embudo.append(("F3 Gini individual > 0.5 (revisar fuga)", len(feats), ", ".join(sospechosas)))
print(f"\nF3 sospechosas de fuga (Gini > 0.5): {sospechosas}")

# ---------------------------------------------------------------- F4 redundancia (correlacion de Spearman)
t = time.time()
num = [c for c in feats if str(tr[c].dtype) != "category"]
mu = tr[num].sample(50_000, random_state=1).rank()          # Spearman = Pearson sobre rangos
corr = mu.corr().abs().values
padre = list(range(len(num)))


def raiz(i):
    while padre[i] != i:
        i = padre[i]
    return i


for i in range(len(num)):
    for j in range(i + 1, len(num)):
        if corr[i, j] > 0.9:
            padre[raiz(j)] = raiz(i)
grupos = pd.Series(num).groupby([raiz(i) for i in range(len(num))]).apply(list)
redund = []
for g in grupos:
    if len(g) > 1:
        mejor = max(g, key=lambda c: gu[c])                  # del grupo, me quedo con la de mas Gini
        redund += [c for c in g if c != mejor]
ejemplo = [g for g in grupos if len(g) > 1][:2]
feats = [c for c in feats if c not in redund]
embudo.append(("F4 redundantes (|Spearman| > 0.9)", len(feats), f"{len(redund)} columnas en {sum(len(g) > 1 for g in grupos)} grupos"))
print(f"\nF4 redundancia ({time.time() - t:.1f} s): {len(redund)} eliminadas. Ejemplos de grupos: {ejemplo}")

# ---------------------------------------------------------------- F5 relevancia: importancia nula (null importance)
t = time.time()
prm = dict(objective="binary", learning_rate=0.1, num_leaves=31, min_child_samples=100, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, verbose=-1, seed=3, num_threads=2)


def ganancia(target):
    m = lgb.train(prm, lgb.Dataset(tr[feats], target), 200)
    return pd.Series(m.feature_importance("gain"), index=feats)


real = ganancia(y)
rng = np.random.default_rng(0)
nulas = pd.concat([ganancia(rng.permutation(y)) for _ in range(8)], axis=1)
umbral = nulas.quantile(0.95, axis=1)
relevantes = real[real > umbral].index.tolist()
fuera = [c for c in feats if c not in relevantes]
print(f"\nF5 importancia nula ({time.time() - t:.0f} s): quedan {len(relevantes)}")
tab = pd.DataFrame({"gain_real": real, "p95_nula": umbral, "gini_univ": gu.reindex(feats)})
tab["queda"] = tab.gain_real > tab.p95_nula
tab = tab.reindex([c for c in ["r03", "s11", "s12", "s13", "s14", "c01", "z01_presente", "t01_dias", "m01", "l03", "n01", "n02", "k05_frec", "k10_frec", "k01"] if c in feats])
print(tab.round(3).to_string())
feats = relevantes
embudo.append(("F5 importancia nula (gana a target barajado)", len(feats), f"{len(fuera)} columnas de ruido"))

print("\nEMBUDO:")
for paso, n, det in embudo:
    print(f"  {paso:<48} -> {n:>4} columnas   {det[:110]}")
sel = pd.Series(feats)
print("\nQue quedo, segun la verdad:", sel.map(lambda c: verdad.get(base(c))).value_counts().to_dict())
pd.Series(feats).to_csv("lab/seleccionadas.csv", index=False)
gu.to_csv("lab/gini_univariado.csv")
print(f"\nTiempo total del embudo: {time.time() - T0:.0f} s")
