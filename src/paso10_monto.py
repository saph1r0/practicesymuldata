"""PASO 10 - Modelo de monto y valor esperado.
Dos preguntas distintas => dos modelos:
  R1: monto esperado de TODA empresa (incluye ceros)  -> lo que mide el R2 en log1p del concurso
  R2: monto SI convierte (solo positivos)              -> para el valor esperado = prob x monto_si_convierte
Por que log1p: el monto tiene cola larga (pocas empresas enormes); en escala log el error de las chicas tambien cuenta."""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from comun import pq, FEAT, SAL

df = pd.read_parquet(pq(FEAT, "tabla_modelo"))
for c in ["seg", "sector", "zona", "tipo_contribuyente", "rating_externo", "calificacion_sbs"]:
    df[c] = df[c].astype("category")
FEATS = [c for c in df.columns if c not in {"k", "target", "monto_target", "es_test"}]
tr, te = df[df.es_test == 0].reset_index(drop=True), df[df.es_test == 1].reset_index(drop=True)
pos = tr[tr.target == 1].reset_index(drop=True)
print("Distribucion del monto entre los que convierten (S/):")
print(pos.monto_target.describe(percentiles=[.5, .9, .99]).round(0).to_string())

prm = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_child_samples=50, verbose=-1, seed=7)


def cv_reg(data, ylog, nombre):
    oof, pred = np.zeros(len(data)), np.zeros(len(te))
    for a, b in KFold(5, shuffle=True, random_state=7).split(data):
        m = lgb.train(prm, lgb.Dataset(data.loc[a, FEATS], ylog[a]), 3000,
                      valid_sets=[lgb.Dataset(data.loc[b, FEATS], ylog[b])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = m.predict(data.loc[b, FEATS]); pred += m.predict(te[FEATS]) / 5
    print(f"  {nombre:<38} R2 (log1p) en validacion = {r2_score(ylog, oof):.3f}")
    return oof, pred


print()
oof_all, pred_all = cv_reg(tr, np.log1p(tr.monto_target.values), "R1 monto de todas las empresas")
oof_pos, pred_pos = cv_reg(pos, np.log1p(pos.monto_target.values), "R2 monto si convierte (solo positivos)")
# alternativa ingenua para comparar: prob x monto medio de los que convierten
prob = pd.read_parquet(f"{SAL}/oof_prob.parquet").oof.values
ingenuo = np.log1p(prob * pos.monto_target.mean())
print(f"  {'Ingenuo: prob x monto promedio':<38} R2 (log1p) en validacion = {r2_score(np.log1p(tr.monto_target), ingenuo):.3f}")

out = pd.DataFrame({"k": te.k, "monto_esperado": np.expm1(pred_all).clip(0), "monto_si_convierte": np.expm1(pred_pos)})
out.to_parquet(f"{SAL}/test_monto.parquet")
print("\n-> salidas/test_monto.parquet")
