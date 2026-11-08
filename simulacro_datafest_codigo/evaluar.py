#!/usr/bin/env python3
"""
Evaluador del simulacro (hace de "jurado tecnico").
Uso: python evaluar.py --entrega entrega.csv --data data_lite

Formato de entrega (una fila por id de ids_evaluacion.csv):
  id_empresa, prob_factoring, monto_esperado, priorizado
    prob_factoring : probabilidad [0,1] de que la empresa haga factoring con BCP (jul-dic 2025)
    monto_esperado : monto en soles esperado de factoring BCP en esa ventana (0 si no)
    priorizado     : 1 si la fuerza de venta la contacta, 0 si no

Metricas:
  1) Gini = 2*AUC - 1 sobre prob_factoring          (eje 2: analitica predictiva)
  2) R2 sobre log1p(monto)                          (eje 2: valor esperado)
  3) Valor neto capturado por los priorizados        (eje 3: asignacion de fuerza de venta)
     ingreso = 1.8% del monto de factoring de las empresas priorizadas que si convierten
     costo   = S/ 150 por empresa contactada
     Restriccion: capacidad total = 2.5% de ids_evaluacion, con cupos por zona.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score

MARGEN = 0.018
COSTO = 150.0
CUPOS = {"LIMA": 0.50, "NORTE": 0.18, "SUR": 0.17, "CENTRO": 0.10, "ORIENTE": 0.05}


def capacidad(n):
    K = int(round(0.025 * n))
    return K, {z: int(K * p) for z, p in CUPOS.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrega", required=True)
    ap.add_argument("--data", default="data_lite")
    a = ap.parse_args()
    clave = pd.read_csv(os.path.join(a.data, "_clave_oculta", "clave.csv"))
    ent = pd.read_csv(a.entrega)
    ent["id_empresa"] = ent["id_empresa"].astype(str).str.strip().str.upper()
    req = {"id_empresa", "prob_factoring", "monto_esperado", "priorizado"}
    if not req.issubset(ent.columns):
        sys.exit(f"Faltan columnas: {req - set(ent.columns)}")
    if ent.id_empresa.duplicated().any():
        sys.exit("ERROR: ids duplicados en la entrega")
    df = clave.merge(ent, on="id_empresa", how="left")
    faltan = df.prob_factoring.isna().sum()
    if faltan:
        print(f"AVISO: {faltan} ids sin prediccion -> se imputan con 0")
    df[["prob_factoring", "monto_esperado", "priorizado"]] = df[["prob_factoring", "monto_esperado", "priorizado"]].fillna(0)

    gini = 2 * roc_auc_score(df.target, df.prob_factoring) - 1
    r2 = r2_score(np.log1p(df.monto_target), np.log1p(df.monto_esperado.clip(lower=0)))

    K, cup = capacidad(len(df))
    pri = df[df.priorizado == 1]
    viol = []
    if len(pri) > K:
        viol.append(f"total {len(pri)} > capacidad {K}")
    for z, c in cup.items():
        nz = (pri.zona == z).sum()
        if nz > c:
            viol.append(f"zona {z}: {nz} > cupo {c}")
    valor = (MARGEN * pri.monto_target).sum() - COSTO * len(pri)
    # oraculo: mejor seleccion posible con los mismos cupos
    df["v"] = MARGEN * df.monto_target - COSTO
    oraculo = sum(df[df.zona == z].nlargest(c, "v").v.clip(lower=0).sum() for z, c in cup.items())
    conv = pri.target.mean() if len(pri) else 0

    print("=" * 60)
    print(" RESULTADO DEL SIMULACRO")
    print("=" * 60)
    print(f" Gini (prob_factoring)          : {gini:.4f}")
    print(f" R2 log1p(monto)                : {r2:.4f}")
    print(f" Priorizados                    : {len(pri)} / capacidad {K}")
    print(f" Tasa de conversion priorizados : {conv:.1%}  (base: {df.target.mean():.1%})")
    print(f" Valor neto capturado (S/)      : {valor:,.0f}")
    print(f" % del oraculo                  : {100 * valor / oraculo:.1f}%")
    if viol:
        print(" !! VIOLACIONES DE CAPACIDAD (en el concurso esto descalifica el eje 3):")
        for v in viol:
            print("    -", v)
    print("-" * 60)
    print(" Gini por sector (para el storytelling):")
    for s, g in df.groupby("sector"):
        if g.target.nunique() == 2 and len(g) > 50:
            print(f"   {s:<26} n={len(g):>6}  tasa={g.target.mean():5.1%}  gini={2 * roc_auc_score(g.target, g.prob_factoring) - 1:.3f}")


if __name__ == "__main__":
    main()
