"""PASO 11 - Habilitar (forma B) y priorizar con capacidad limitada.
1) score = criterio de valor del equipo (aqui: Pyme primero, ver CRITERIO)
2) habilitar: entre empresas SIN lead, las de mayor score, hasta duplicar los leads
3) priorizar: entre leads + habilitados, por zona, las de mayor score hasta llenar cada cupo
Con un solo recurso y cupos independientes por zona, ordenar y cortar (voraz) da el optimo."""
import sys
import numpy as np
import pandas as pd
from comun import pq, FEAT, SAL, CUPOS

CRITERIO = sys.argv[1] if len(sys.argv) > 1 else "pyme"      # "pyme" | "soles" | "conversiones"
df = pd.read_parquet(pq(FEAT, "tabla_modelo"), columns=["k", "seg", "zona", "lead", "es_test"])
te = (df[df.es_test == 1].merge(pd.read_parquet(f"{SAL}/test_prob.parquet"), on="k")
      .merge(pd.read_parquet(f"{SAL}/test_monto.parquet"), on="k"))
te["valor_soles"] = te.prob * te.monto_si_convierte * 0.018 - 150     # ingreso esperado - costo de visita
es_pyme = te.seg.isin(["MIPE", "PYME"])
if CRITERIO == "soles":
    te["score"], elegible = te.valor_soles, ~te.seg.isin(["ESTADO", "CORPORATIVO"])
elif CRITERIO == "conversiones":
    te["score"], elegible = te.prob, es_pyme
else:  # pyme: valor en soles, pero solo Pyme
    te["score"], elegible = te.valor_soles, es_pyme

n_leads = int(te.lead.sum())
te["habilitar"] = 0
te.loc[te[(te.lead == 0) & elegible].nlargest(n_leads, "score").index, "habilitar"] = 1

K = int(round(0.025 * len(te)))
te["priorizado"] = 0
cand = te[((te.lead == 1) | (te.habilitar == 1)) & elegible & (te.zona != "DESCONOCIDA")]
for z, p in CUPOS.items():
    te.loc[cand[cand.zona == z].nlargest(int(K * p), "score").index, "priorizado"] = 1
print(f"criterio={CRITERIO} | leads hoy {n_leads:,} | habilitados {te.habilitar.sum():,} | "
      f"priorizados {te.priorizado.sum():,} de {K:,} (A: {int((te.priorizado * te.lead).sum()):,})")
print(te[te.priorizado == 1].groupby("zona").size().rename("priorizados").to_string())

te["id_empresa"] = "EMP" + te.k.astype(str).str.zfill(8)
te = te.rename(columns={"prob": "prob_factoring"})
te[["id_empresa", "prob_factoring", "monto_esperado", "habilitar", "priorizado"]].to_csv(f"{SAL}/entrega_{CRITERIO}.csv", index=False)
print(f"-> salidas/entrega_{CRITERIO}.csv")
