"""PASO 9 - Modelo de probabilidad: de lo mas simple a lo mas complejo, y solo se queda lo que gana.
  M0 regla de negocio (sin ML)  -> el piso: si el ML no le gana, algo esta mal
  M1 regresion logistica        -> lineal, explicable, rapida
  M2 LightGBM                   -> arboles con boosting: no linealidades, nulos y categorias nativos
Validacion: 5-fold estratificado sobre TRAIN. Metrica: Gini global y dentro de los grupos A y B."""
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from comun import pq, FEAT, SAL

df = pd.read_parquet(pq(FEAT, "tabla_modelo"))
CATS = ["seg", "sector", "zona", "tipo_contribuyente", "rating_externo", "calificacion_sbs"]
for c in CATS:
    df[c] = df[c].astype("category")
NO_FEAT = {"k", "target", "monto_target", "es_test"}
FEATS = [c for c in df.columns if c not in NO_FEAT]
tr = df[df.es_test == 0].reset_index(drop=True)
te = df[df.es_test == 1].reset_index(drop=True)
y = tr.target.values
grupoA = (tr.lead == 1).values
grupoB = ((tr.lead == 0) & tr.seg.isin(["MIPE", "PYME"]) & (tr.cliente == 1)).values
skf = list(StratifiedKFold(5, shuffle=True, random_state=7).split(tr, y))
gini = lambda yy, p: 2 * roc_auc_score(yy, p) - 1


def reporte(nombre, oof, seg):
    print(f"  {nombre:<26} Gini global {gini(y, oof):.3f} | A (leads) {gini(y[grupoA], oof[grupoA]):.3f} | "
          f"B (Pyme sin lead) {gini(y[grupoB], oof[grupoB]):.3f} | {seg:5.0f} s")


print(f"train {len(tr):,} | test {len(te):,} | variables {len(FEATS)} | tasa {y.mean():.2%}\n")

# ---------- M0: regla de negocio escrita a mano con las hipotesis del EDA
t = time.time()
regla = (2 * tr.lead + (tr.plazo_prom.fillna(0) >= 60) + tr.share_grandes.fillna(0)
         + 0.2 * np.log1p(tr.vol_facturas) + (tr.n_ops_bcp > 0)).values
reporte("M0 regla de negocio", regla, time.time() - t)

# ---------- M1: regresion logistica (necesita imputar, escalar y one-hot: no acepta nulos ni texto)
t = time.time()
num = ["lead", "cliente", "log_vol_facturas", "plazo_prom", "share_grandes", "share_plazo60", "n_pagadores",
       "n_ops_bcp", "n_ops_otro", "uso_linea_pct_rec3", "saldo_cta_cte_tend", "score_buro", "activo"]
pre = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
    (OneHotEncoder(handle_unknown="ignore"), ["sector", "seg", "zona"]))
oof1 = np.zeros(len(tr))
for a, b in skf:
    mdl = make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))
    X = tr[num + ["sector", "seg", "zona"]].astype({c: str for c in ["sector", "seg", "zona"]})
    mdl.fit(X.iloc[a], y[a])
    oof1[b] = mdl.predict_proba(X.iloc[b])[:, 1]
reporte("M1 regresion logistica", oof1, time.time() - t)

# ---------- M2: LightGBM con early stopping (cada fold decide cuantos arboles necesita)
t = time.time()
params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
oof2, pred_te, imp, iters = np.zeros(len(tr)), np.zeros(len(te)), pd.Series(0.0, index=FEATS), []
for a, b in skf:
    m = lgb.train(params, lgb.Dataset(tr.loc[a, FEATS], y[a]), num_boost_round=3000,
                  valid_sets=[lgb.Dataset(tr.loc[b, FEATS], y[b])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    oof2[b] = m.predict(tr.loc[b, FEATS])
    pred_te += m.predict(te[FEATS]) / len(skf)          # promedio de los 5 modelos = prediccion mas estable
    imp += pd.Series(m.feature_importance("gain"), index=FEATS)
    iters.append(m.best_iteration)
reporte("M2 LightGBM", oof2, time.time() - t)
print(f"     arboles por fold: {iters}")

# ---------- estabilidad: el Gini por fold no deberia variar mucho
print("     Gini por fold:", [round(gini(y[b], oof2[b]), 3) for _, b in skf])

# ---------- tabla de lift por decil: lo que entiende el negocio
d = pd.DataFrame({"p": oof2, "y": y, "m": tr.monto_target})
d["decil"] = pd.qcut(d.p.rank(method="first", ascending=False), 10, labels=range(1, 11))
lift = d.groupby("decil", observed=True).agg(empresas=("y", "size"), conversiones=("y", "sum"), tasa=("y", "mean"))
lift["% del total capturado (acum)"] = (lift.conversiones.cumsum() / lift.conversiones.sum() * 100).round(1)
lift["lift"] = (lift.tasa / y.mean()).round(1)
print("\nLift por decil (decil 1 = el 10% con mayor probabilidad):\n" + lift.round(3).to_string())

imp = (imp / imp.sum()).sort_values(ascending=False)
imp.to_csv(f"{SAL}/importancia.csv")
pd.DataFrame({"k": tr.k, "oof": oof2}).to_parquet(f"{SAL}/oof_prob.parquet")
pd.DataFrame({"k": te.k, "prob": pred_te}).to_parquet(f"{SAL}/test_prob.parquet")
print("\nTop 12 variables (ganancia):\n" + imp.head(12).round(3).to_string())
