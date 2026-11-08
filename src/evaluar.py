#!/usr/bin/env python3
"""
Evaluador del simulacro (hace de "jurado tecnico").
Uso: python evaluar.py --entrega entrega.csv --data data_lite

Formato de entrega (una fila por id de ids_evaluacion.csv):
  id_empresa, prob_factoring, monto_esperado, habilitar, priorizado
    prob_factoring : probabilidad [0,1] de factoring con BCP (jul-dic 2025)
    monto_esperado : monto en soles esperado de factoring BCP en esa ventana
    habilitar      : 1 si proponen darle lead a una empresa que HOY NO lo tiene (forma B: "habilitar mas")
    priorizado     : 1 si la fuerza comercial la contacta en la campana

Reglas (como en la lamina "Reto del caso"):
  - Grupo A = empresas con lead hoy (flag_lead_factoring, ~10% de clientes Pyme). Forma A: "convertir mas".
  - Forma B: pueden habilitar como maximo tantas empresas nuevas como leads hay hoy (de 10% a 20%).
  - Solo se puede priorizar a quien tiene lead (actual o habilitado por ustedes).
  - Capacidad comercial: 2.5% de ids_evaluacion, con cupos por zona.
  - El criterio de "cliente mas valioso" lo define cada equipo; aqui se reportan varias medidas.
    El valor neto usa un supuesto de referencia: 1.8% del monto - S/ 150 por contacto.
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


def gini(y, p):
    return 2 * roc_auc_score(y, p) - 1 if y.nunique() == 2 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entrega", required=True)
    ap.add_argument("--data", default="data_lite")
    a = ap.parse_args()
    clave = pd.read_csv(os.path.join(a.data, "_clave_oculta", "clave.csv"))
    ent = pd.read_csv(a.entrega)
    ent["id_empresa"] = ent["id_empresa"].astype(str).str.strip().str.upper()
    req = {"id_empresa", "prob_factoring", "monto_esperado", "habilitar", "priorizado"}
    if not req.issubset(ent.columns):
        sys.exit(f"Faltan columnas: {req - set(ent.columns)}")
    if ent.id_empresa.duplicated().any():
        sys.exit("ERROR: ids duplicados en la entrega")
    df = clave.merge(ent, on="id_empresa", how="left")
    faltan = df.prob_factoring.isna().sum()
    if faltan:
        print(f"AVISO: {faltan} ids sin prediccion -> se imputan con 0")
    cols = ["prob_factoring", "monto_esperado", "habilitar", "priorizado"]
    df[cols] = df[cols].fillna(0)
    viol = []

    # ---- habilitar (forma B) ----
    n_leads = int(df.lead.sum())
    if ((df.habilitar == 1) & (df.lead == 1)).any():
        print("AVISO: marcaron habilitar=1 en empresas que ya tienen lead (se ignora)")
    df["hab"] = ((df.habilitar == 1) & (df.lead == 0)).astype(int)
    if df.hab.sum() > n_leads:
        viol.append(f"habilitaron {df.hab.sum()} > maximo {n_leads} (duplicar la base de leads)")
    df["con_lead"] = ((df.lead == 1) | (df.hab == 1)).astype(int)

    # ---- priorizar ----
    K, cup = capacidad(len(df))
    pri = df[df.priorizado == 1]
    if (pri.con_lead == 0).any():
        viol.append(f"{int((pri.con_lead == 0).sum())} priorizados sin lead (ni actual ni habilitado)")
    if len(pri) > K:
        viol.append(f"total priorizados {len(pri)} > capacidad {K}")
    for z, c in cup.items():
        nz = (pri.zona == z).sum()
        if nz > c:
            viol.append(f"zona {z}: {nz} > cupo {c}")

    A = df[df.grupo == "A_LEAD"]
    B = df[df.grupo == "B_PYME_SIN_LEAD"]
    NL = df[df.lead == 0]
    hab = df[df.hab == 1]
    orac_hab = NL.nlargest(n_leads, "monto_target")
    df["v"] = MARGEN * df.monto_target - COSTO
    oraculo = sum(df[df.zona == z].nlargest(c, "v").v.clip(lower=0).sum() for z, c in cup.items())
    valor = (MARGEN * pri.monto_target).sum() - COSTO * len(pri)

    print("=" * 66)
    print(" RESULTADO DEL SIMULACRO")
    print("=" * 66)
    print(" EJE 2 - VALORAR")
    print(f"   Gini global                          : {gini(df.target, df.prob_factoring):.4f}")
    print(f"   Gini dentro de A (con lead)          : {gini(A.target, A.prob_factoring):.4f}   <- convertir mas")
    print(f"   Gini dentro de B (Pyme sin lead)     : {gini(B.target, B.prob_factoring):.4f}   <- habilitar mas")
    print(f"   R2 log1p(monto)                      : {r2_score(np.log1p(df.monto_target), np.log1p(df.monto_esperado.clip(lower=0))):.4f}")
    print(" EJE 1 - HABILITAR (forma B)")
    print(f"   Leads hoy en evaluacion              : {n_leads}  (tasa {A.target.mean():.1%})")
    print(f"   Habilitados por ustedes              : {len(hab)} (maximo {n_leads})")
    if len(hab):
        print(f"   Tasa de conversion habilitados       : {hab.target.mean():.1%}  (base sin lead: {NL.target.mean():.1%}, lift {hab.target.mean() / max(NL.target.mean(), 1e-9):.1f}x)")
        print(f"   Monto potencial habilitado (S/)      : {hab.monto_target.sum():,.0f}  ({100 * hab.monto_target.sum() / orac_hab.monto_target.sum():.1f}% del optimo)")
        print(f"   Mezcla de habilitados por grupo      : {hab.grupo.value_counts().to_dict()}")
    print(" EJE 3 - PRIORIZAR")
    print(f"   Priorizados                          : {len(pri)} / capacidad {K}  -> de A: {(pri.lead == 1).sum()}, habilitados: {(pri.hab == 1).sum()}")
    print(f"   Conversiones esperables              : {int(pri.target.sum())}  (tasa {pri.target.mean() if len(pri) else 0:.1%}, base {df.target.mean():.1%})")
    print(f"   Monto de factoring capturado (S/)    : {pri.monto_target.sum():,.0f}")
    print(f"   Valor neto, criterio de referencia   : {valor:,.0f}  ({100 * valor / oraculo:.1f}% del optimo)")
    print("   (El jurado evalua SU criterio de 'cliente mas valioso'; esta cifra es solo una referencia comun.)")
    if viol:
        print(" !! VIOLACIONES (invalidan el eje):")
        for v in viol:
            print("    -", v)
    print("-" * 66)
    print(" Gini por sector (para el storytelling):")
    for s, g in df.groupby("sector"):
        if g.target.nunique() == 2 and len(g) > 50:
            print(f"   {s:<26} n={len(g):>6}  tasa={g.target.mean():5.1%}  gini={gini(g.target, g.prob_factoring):.3f}")


if __name__ == "__main__":
    main()
