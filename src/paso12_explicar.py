"""PASO 12 - Explicar el modelo.
  a) Importancia agrupada por FUENTE de datos: que tabla aporto mas
  b) Contribuciones SHAP (LightGBM las calcula solo con pred_contrib=True): por que UNA empresa tiene score alto
  c) Gini por rubro: donde el modelo es confiable y donde no"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from comun import pq, FEAT, SAL

df = pd.read_parquet(pq(FEAT, "tabla_modelo"))
for c in ["seg", "sector", "zona", "tipo_contribuyente", "rating_externo", "calificacion_sbs"]:
    df[c] = df[c].astype("category")
FEATS = [c for c in df.columns if c not in {"k", "target", "monto_target", "es_test"}]
tr = df[df.es_test == 0].reset_index(drop=True)

# a) importancia por fuente
imp = pd.read_csv(f"{SAL}/importancia.csv", index_col=0).iloc[:, 0]
fuente = {}
for f in ["f_maestro", "f_red", "f_pagador", "f_transf", "f_factoring", "f_perfil"]:
    for c in pd.read_parquet(pq(FEAT, f)).columns:
        fuente[c] = f
for c in ["log_vol_facturas", "crec_facturas", "facturas_sobre_ventas", "es_pagador"]:
    fuente[c] = "f_red"
fuente["ya_factoriza_otro"] = "f_factoring"
print("a) Importancia por fuente de datos (% de la ganancia total):")
print((imp.groupby(imp.index.map(fuente)).sum() * 100).sort_values(ascending=False).round(1).to_string())

# b) SHAP de un modelo final entrenado con todo train
m = lgb.train(dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100, feature_fraction=0.8,
                   bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7),
              lgb.Dataset(tr[FEATS], tr.target), 250)
muestra = tr.sample(20000, random_state=1)
contrib = pd.DataFrame(m.predict(muestra[FEATS], pred_contrib=True)[:, :-1], columns=FEATS, index=muestra.index)
print("\nb) SHAP medio absoluto (cuanto mueve cada variable el score, en log-odds):")
print(contrib.abs().mean().sort_values(ascending=False).head(10).round(3).to_string())
ej = muestra[(muestra.lead == 0) & (muestra.seg == "PYME") & (muestra.n_ops_bcp == 0)].assign(
    p=lambda d: m.predict(d[FEATS])).nlargest(1, "p")
i = ej.index[0]
print(f"\n   Ejemplo: empresa k={int(ej.k.iloc[0])} (PYME, sin lead, nunca hizo factoring BCP), "
      f"sector {ej.sector.iloc[0]}, prob {ej.p.iloc[0]:.0%}, convirtio realmente: {int(ej.target.iloc[0])}")
top = contrib.loc[i].sort_values(key=abs, ascending=False).head(6)
for c, v in top.items():
    print(f"     {c:<26} valor={muestra.loc[i, c]!s:<14} aporte={v:+.2f}")

# c) Gini por rubro (validacion fuera de muestra)
oof = pd.read_parquet(f"{SAL}/oof_prob.parquet")
x = tr[["k", "sector", "target"]].merge(oof, on="k")
print("\nc) Gini por rubro (validacion cruzada):")
r = x.groupby("sector", observed=True).apply(
    lambda g: pd.Series({"empresas": len(g), "tasa_%": 100 * g.target.mean(),
                         "gini": 2 * roc_auc_score(g.target, g.oof) - 1 if g.target.nunique() > 1 else np.nan}),
    include_groups=False)
print(r.sort_values("gini").round(3).to_string())
