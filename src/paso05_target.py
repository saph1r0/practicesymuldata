"""PASO 5 - Linea de tiempo, target y separacion train/test.
Idea central: el modelo se usara el 30/06/2025 para predecir jul-dic 2025.
=> toda variable debe calcularse con informacion <= 2025-06-30; el target vive en 2025-07-01..2025-12-31."""
import pandas as pd
from sklearn.metrics import roc_auc_score
from comun import csv, pq, LIMPIO, FEAT, CORTE, INI_TGT, FIN_TGT, conectar

con = conectar()
F = pq(LIMPIO, "factoring")

print("1) Que fechas hay en cada tabla (la linea de tiempo)")
print("   transacciones:", con.execute(f"SELECT min(fh)::DATE, max(fh)::DATE FROM '{pq(LIMPIO, 'transacciones')}'").fetchone())
print("   factoring    :", con.execute(f"SELECT min(fecha_desembolso), max(fecha_desembolso) FROM '{F}'").fetchone())
print("   => factoring llega hasta dic-2025: es la UNICA tabla que contiene el futuro (el target).")

print("\n2) Target: >=1 operacion BCP no anulada en la ventana")
con.execute(f"""
CREATE TABLE tgt AS
SELECT k, 1 AS target, sum(monto_pen) AS monto_target, count(*) AS n_ops
FROM '{F}' WHERE banco = 'BCP' AND estado <> 'ANULADO'
  AND fecha_desembolso BETWEEN '{INI_TGT}' AND '{FIN_TGT}' GROUP BY k""")

print("\n3) Separacion: test = ids_evaluacion (no tienen target visible); train = el resto")
con.execute(f"CREATE TABLE test_ids AS SELECT k(id_empresa) AS k FROM read_csv('{csv('ids_evaluacion.csv')}', all_varchar=true)")
base = con.execute(f"""
  SELECT m.k, m.seg, m.lead, (t.k IS NOT NULL)::INT AS es_test,
         coalesce(g.target, 0) AS target, coalesce(g.monto_target, 0) AS monto_target
  FROM '{pq(LIMPIO, 'maestro')}' m LEFT JOIN test_ids t USING (k) LEFT JOIN tgt g USING (k)""").df()
tr = base[base.es_test == 0]
print(f"   train {len(tr):,} empresas, tasa target {tr.target.mean():.2%} | test {base.es_test.sum():,} empresas")
print(f"   en test, empresas con target visible (debe ser 0): {base[(base.es_test == 1) & (base.target == 1)].shape[0]}")
print("   tasa por grupo (train):")
print(tr.assign(grupo=tr.lead.map({1: "A: con lead"}).fillna(tr.seg.isin(["MIPE", "PYME"]).map({True: "B: Pyme sin lead", False: "C: resto"})))
        .groupby("grupo").agg(empresas=("target", "size"), tasa=("target", "mean"), monto_medio_si_convierte=("monto_target", lambda x: x[x > 0].mean()))
        .round(3).to_string())
base[["k", "es_test", "target", "monto_target"]].to_parquet(pq(FEAT, "target"), index=False)

print("\n4) DEMOSTRACION DE FUGA: contar operaciones BCP con y sin filtro de fecha")
fuga = con.execute(f"""
  SELECT k, count(*) FILTER (WHERE fecha_desembolso <= '{CORTE}') AS ops_ok,
            count(*) AS ops_fuga
  FROM '{F}' WHERE banco = 'BCP' AND estado <> 'ANULADO' GROUP BY k""").df()
x = base.merge(fuga, on="k", how="left").fillna(0)
for c in ["ops_ok", "ops_fuga"]:
    g_tr = 2 * roc_auc_score(x[x.es_test == 0].target, x[x.es_test == 0][c]) - 1
    print(f"   {c:<9} Gini en train = {g_tr:.3f}   media en train = {x[x.es_test == 0][c].mean():.3f}   media en test = {x[x.es_test == 1][c].mean():.3f}")
print("   ops_fuga 'se ve' genial en train, pero en test la ventana fue borrada: el modelo fallaria en el set ciego.")
