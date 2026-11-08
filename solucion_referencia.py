#!/usr/bin/env python3
"""
SOLUCION DE REFERENCIA del simulacro (NO LA ABRAN hasta terminar su intento de 10h).
Stack: DuckDB (limpieza + features con SQL, aguanta 8M filas en laptop) + LightGBM.
Uso: python solucion_referencia.py --data data_lite --salida entrega_ref.csv

Secciones = una por integrante en el concurso:
  A. limpieza_maestro()        -> Data Engineer
  B. features_transacciones()  -> Data Engineer / Analytics Engineer (red pagador-proveedor)
  C. features_factoring()      -> Analytics Engineer (+ construccion del target)
  D. features_perfil()         -> BI/negocio (detecta fugas)
  E. modelar()                 -> Data Scientist
  F. asignar()                 -> Estratega de negocio (optimizacion con cupos)
"""
import argparse
import os
import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TC = 3.75
CUPOS = {"LIMA": 0.50, "NORTE": 0.18, "SUR": 0.17, "CENTRO": 0.10, "ORIENTE": 0.05}
ZONA = {"LIMA": "LIMA", "CALLAO": "LIMA", "AREQUIPA": "SUR", "LA LIBERTAD": "NORTE", "PIURA": "NORTE",
        "LAMBAYEQUE": "NORTE", "CUSCO": "SUR", "JUNIN": "CENTRO", "ICA": "SUR", "ANCASH": "NORTE",
        "CAJAMARCA": "NORTE", "PUNO": "SUR", "LORETO": "ORIENTE", "SAN MARTIN": "ORIENTE", "TACNA": "SUR",
        "UCAYALI": "ORIENTE", "HUANUCO": "CENTRO", "MOQUEGUA": "SUR"}
DEP_FIX = {"AQP": "AREQUIPA", "LIMA METROPOLITANA": "LIMA", "CUZCO": "CUSCO", "LALIBERTAD": "LA LIBERTAD",
           "PROV. CONST. DEL CALLAO": "CALLAO", "JUNÃ\x8dN": "JUNIN"}
SEG_FIX = {"MICRO": "MIPE", "MYPE": "MIPE", "PEQUEÑA EMPRESA": "PYME", "MEDIANA EMPRESA": "MEDIANA",
           "GRAN EMPRESA": "GRANDE", "CORP": "CORPORATIVO", "GOBIERNO": "ESTADO", "SECTOR PUBLICO": "ESTADO"}


def key_sql(col):
    """id texto -> entero comparable entre tablas (quita EMP, ceros, espacios, mayusculas)."""
    return f"TRY_CAST(regexp_replace(upper(trim({col})), '^EMP0*', '') AS BIGINT)"


def quitar_tildes(s):
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


# ---------------------------------------------------------------- A
def limpieza_maestro(con, d):
    con.execute(f"""
    CREATE OR REPLACE TABLE m_raw AS
    SELECT *, {key_sql('id_empresa')} AS k FROM read_csv('{d}/maestro_empresas.csv', all_varchar=true);
    CREATE OR REPLACE TABLE m AS
    SELECT * EXCLUDE (rn) FROM (
      SELECT *, row_number() OVER (PARTITION BY k ORDER BY fecha_actualizacion DESC) rn FROM m_raw) WHERE rn = 1;
    """)
    m = con.execute("SELECT * FROM m").df()
    # segmento
    s = m.segmento.fillna("").str.upper().str.strip()
    m["seg"] = s.map(lambda x: SEG_FIX.get(x, x))
    # sector: ciiu normalizado (fuente mas confiable) -> sector mayoritario
    m["ciiu"] = m.ciiu_rev4.fillna("").str.replace(".0", "", regex=False).str.zfill(4).replace("0000", np.nan)
    sec_lim = m.sector_economico_sbs  # columna limpia pero con 30% faltante
    mapa = pd.DataFrame({"c": m.ciiu, "s": sec_lim}).dropna().groupby("c").s.agg(lambda x: x.mode()[0])
    m["sector"] = m.ciiu.map(mapa).fillna(sec_lim).fillna("DESCONOCIDO")
    # departamento -> zona
    dep = m.departamento.fillna("").map(lambda x: quitar_tildes(x.upper().strip()) if x else "")
    dep = dep.map(lambda x: DEP_FIX.get(x, x))
    m["dep"] = dep.replace("", np.nan)
    m["zona"] = m.dep.map(ZONA).fillna(m.zona_comercial).fillna("LIMA")
    m["cliente"] = m.flag_cliente_bcp.fillna("").str.upper().isin(["S", "SI", "1", "Y"]).astype(int)
    v = pd.to_numeric(m.ventas_anuales_soles, errors="coerce")
    m["ventas"] = v.where((v > 0) & (v != 99999999))
    nt = pd.to_numeric(m.num_trabajadores, errors="coerce")
    m["ntrab"] = nt.where(nt >= 0)
    fc = pd.to_datetime(m.fecha_constitucion, errors="coerce", format="mixed", dayfirst=True)
    fc = fc.where(fc.dt.year > 1900)
    m["antig_anios"] = (pd.Timestamp("2025-06-30") - fc).dt.days / 365
    m["activo"] = (m.estado_contribuyente == "ACTIVO").astype(int)
    for c in ["flag_exportador", "flag_importador", "flag_agente_retencion", "flag_buen_contribuyente", "num_establecimientos", "score_buro", "antiguedad_cliente_meses"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["habido"] = (m.condicion_domicilio == "HABIDO").astype(int)
    keep = ["k", "id_empresa", "seg", "sector", "ciiu", "zona", "cliente", "ventas", "ntrab", "antig_anios", "activo", "habido",
            "flag_exportador", "flag_importador", "flag_agente_retencion", "flag_buen_contribuyente",
            "num_establecimientos", "score_buro", "antiguedad_cliente_meses", "rating_externo", "tipo_contribuyente"]
    m = m[keep]
    con.register("m_df", m)
    con.execute("CREATE OR REPLACE TABLE mc AS SELECT * FROM m_df")
    return m


# ---------------------------------------------------------------- B
def features_transacciones(con, d):
    con.execute(f"""
    CREATE OR REPLACE TABLE txc AS
    WITH raw AS (SELECT DISTINCT * FROM read_csv('{d}/transacciones.csv', all_varchar=true))
    SELECT {key_sql('id_ordenante')} AS ord, {key_sql('id_beneficiario')} AS ben,
      COALESCE(try_strptime(fecha_hora, '%Y-%m-%d %H:%M:%S'), try_strptime(fecha_hora, '%d/%m/%Y %H:%M')) AS fh,
      tipo_transaccion AS tipo,
      TRY_CAST(replace(monto, ',', '') AS DOUBLE) * CASE WHEN upper(trim(moneda)) IN ('USD','US$') THEN {TC} ELSE 1 END AS monto,
      TRY_CAST(dias_credito_documento AS DOUBLE) AS dcred
    FROM raw WHERE estado_tx = 'PROCESADA';
    DELETE FROM txc WHERE monto IS NULL OR monto <= 0 OR monto > 5e8;

    -- red pagador -> proveedor: solo pagos de facturas
    CREATE OR REPLACE TABLE fx AS
    SELECT t.*, p.seg AS seg_pag, p.sector AS sec_pag FROM txc t LEFT JOIN mc p ON t.ord = p.k
    WHERE tipo IN ('PAGO_FACTURA', 'PAGO_PROVEEDORES_MASIVO');

    -- plazo tipico de cada pagador (para imputar dcred faltante)
    CREATE OR REPLACE TABLE plz_pag AS SELECT ord, median(dcred) AS plz_pag FROM fx GROUP BY ord;

    CREATE OR REPLACE TABLE f_tx AS
    SELECT ben AS k,
      count(*) AS n_fact, sum(monto) AS vol_fact, count(DISTINCT ord) AS n_pagadores,
      avg(monto) AS ticket_prom,
      avg(coalesce(dcred, pp.plz_pag)) AS plazo_prom, max(coalesce(dcred, pp.plz_pag)) AS plazo_max,
      sum(CASE WHEN coalesce(dcred, pp.plz_pag) >= 60 THEN monto ELSE 0 END) / sum(monto) AS share_plazo60,
      sum(CASE WHEN seg_pag IN ('GRANDE','CORPORATIVO','ESTADO') THEN monto ELSE 0 END) / sum(monto) AS share_grandes,
      sum(CASE WHEN seg_pag = 'ESTADO' THEN monto ELSE 0 END) AS vol_estado,
      sum(CASE WHEN fh >= TIMESTAMP '2025-01-01' THEN monto ELSE 0 END) AS vol_ult6m,
      sum(CASE WHEN fh < TIMESTAMP '2025-01-01' AND fh >= TIMESTAMP '2024-07-01' THEN monto ELSE 0 END) AS vol_prev6m,
      count(DISTINCT date_trunc('month', fh)) AS meses_activos,
      date_diff('day', max(fh), TIMESTAMP '2025-06-30') AS dias_ult_fact
    FROM fx LEFT JOIN plz_pag pp USING (ord) GROUP BY ben;

    CREATE OR REPLACE TABLE f_pag AS
    SELECT ord AS k, sum(monto) AS vol_pagado_fact, count(DISTINCT ben) AS n_proveedores FROM fx GROUP BY ord;
    CREATE OR REPLACE TABLE f_tr AS
    SELECT k, sum(n_in) AS n_transf_in, sum(n_out) AS n_transf_out FROM (
      SELECT ben AS k, count(*) n_in, 0 n_out FROM txc WHERE tipo LIKE 'TRANSF%' GROUP BY ben
      UNION ALL SELECT ord, 0, count(*) FROM txc WHERE tipo LIKE 'TRANSF%' GROUP BY ord) GROUP BY k;
    """)


# ---------------------------------------------------------------- C
def features_factoring(con, d):
    con.execute(f"""
    CREATE OR REPLACE TABLE fac AS
    SELECT {key_sql('id_proveedor')} AS k, {key_sql('id_pagador')} AS kp,
      CASE WHEN entidad_financiera IN ('BCP','002') THEN 'BCP' ELSE 'OTRO' END AS banco,
      fuente_registro, estado_operacion AS est, CAST(fecha_desembolso AS DATE) AS fdes,
      TRY_CAST(monto_facturas AS DOUBLE) * CASE WHEN moneda = 'USD' THEN {TC} ELSE 1 END AS monto,
      TRY_CAST(replace(tasa_mensual, '%', '') AS DOUBLE) AS tasa
    FROM read_csv('{d}/factoring_operaciones.csv', all_varchar=true)
    -- las filas SPI con codigo 002 son la MISMA operacion BCP vista por el SPI: se eliminan
    WHERE NOT (fuente_registro = 'SPI' AND entidad_financiera = '002');

    CREATE OR REPLACE TABLE f_fac AS
    SELECT k,
      count(*) FILTER (WHERE banco='BCP') AS n_ops_bcp, count(*) FILTER (WHERE banco='OTRO') AS n_ops_otro,
      sum(monto) FILTER (WHERE banco='BCP') AS monto_bcp_hist, sum(monto) FILTER (WHERE banco='OTRO') AS monto_otro_hist,
      date_diff('day', max(fdes), DATE '2025-06-30') AS dias_ult_fact_op,
      avg(tasa) AS tasa_prom_hist, count(*) FILTER (WHERE est='VENCIDO') AS n_vencidas
    FROM fac WHERE fdes < DATE '2025-07-01' AND est <> 'ANULADO' GROUP BY k;

    -- TARGET (solo existe para empresas de train)
    CREATE OR REPLACE TABLE tgt AS
    SELECT k, 1 AS target, sum(monto) AS monto_target FROM fac
    WHERE banco='BCP' AND est <> 'ANULADO' AND fdes BETWEEN DATE '2025-07-01' AND DATE '2025-12-31' GROUP BY k;
    """)


# ---------------------------------------------------------------- D
def features_perfil(con, d):
    p = con.execute(f"SELECT * FROM read_csv('{d}/perfil_empresa.csv', all_varchar=true)").df()
    p["k"] = pd.to_numeric(p.id_empresa.str.replace("EMP", "", regex=False), errors="coerce")
    # FUGAS: informacion posterior al corte -> se eliminan
    p = p.drop(columns=["flag_factoring_activo", "fecha_ultimo_contacto_comercial", "resultado_ultimo_contacto", "periodo_corte", "id_empresa"])
    num = [c for c in p.columns if c not in ("k", "calificacion_sbs")]
    p[num] = p[num].apply(pd.to_numeric, errors="coerce").replace(-999, np.nan)
    out = p[["k", "calificacion_sbs", "num_productos_activos", "flag_cta_cte", "flag_prestamo_capital_trabajo", "flag_leasing",
             "flag_carta_fianza", "flag_telecredito", "score_riesgo_interno", "num_entidades_sistema", "linea_credito_aprobada"]].copy()
    for v in ["saldo_cta_cte", "uso_linea_pct", "deuda_sistema", "monto_abonos", "dias_atraso_max", "num_tx_abono"]:
        rec = p[[f"{v}_m{j:02d}" for j in range(1, 4)]].mean(axis=1)      # ultimos 3 meses
        ant = p[[f"{v}_m{j:02d}" for j in range(10, 13)]].mean(axis=1)    # hace 10-12 meses
        out[f"{v}_rec3"] = rec
        out[f"{v}_tend"] = (rec - ant) / (ant.abs() + 1)
        out[f"{v}_std"] = p[[f"{v}_m{j:02d}" for j in range(1, 13)]].std(axis=1)
    out["atraso_max12"] = p[[f"dias_atraso_max_m{j:02d}" for j in range(1, 13)]].max(axis=1)
    con.register("p_df", out)
    con.execute("CREATE OR REPLACE TABLE f_per AS SELECT * FROM p_df")


# ---------------------------------------------------------------- E
def modelar(con, d):
    test_ids = pd.read_csv(f"{d}/ids_evaluacion.csv").id_empresa
    con.register("t_df", pd.DataFrame({"k": test_ids.str.replace("EMP", "").astype(int)}))
    df = con.execute("""
      SELECT mc.*, f_tx.* EXCLUDE (k), f_pag.* EXCLUDE (k), f_tr.* EXCLUDE (k), f_fac.* EXCLUDE (k), f_per.* EXCLUDE (k),
             coalesce(tgt.target, 0) AS target, coalesce(tgt.monto_target, 0) AS monto_target,
             (t_df.k IS NOT NULL) AS es_test
      FROM mc LEFT JOIN f_tx USING (k) LEFT JOIN f_pag USING (k) LEFT JOIN f_tr USING (k)
      LEFT JOIN f_fac USING (k) LEFT JOIN f_per USING (k) LEFT JOIN tgt USING (k) LEFT JOIN t_df USING (k)
    """).df()
    for c in ["n_fact", "vol_fact", "n_pagadores", "vol_estado", "n_ops_bcp", "n_ops_otro", "vol_pagado_fact"]:
        df[c] = df[c].fillna(0)
    df["log_vol_fact"] = np.log1p(df.vol_fact)
    df["crec_fact"] = (df.vol_ult6m - df.vol_prev6m) / (df.vol_prev6m + 1)
    df["fact_sobre_ventas"] = df.vol_fact / 1.5 / df.ventas
    df["es_pagador"] = (df.vol_pagado_fact > df.vol_fact).astype(int)
    cats = ["seg", "sector", "zona", "rating_externo", "tipo_contribuyente", "calificacion_sbs"]
    for c in cats:
        df[c] = df[c].astype("category")
    drop = {"k", "id_empresa", "ciiu", "target", "monto_target", "es_test"}
    feats = [c for c in df.columns if c not in drop]
    tr, te = df[~df.es_test].reset_index(drop=True), df[df.es_test].reset_index(drop=True)
    print(f"train={len(tr):,} (tasa {tr.target.mean():.2%}) | test={len(te):,} | features={len(feats)}")

    pc = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=50, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=7)
    oof = np.zeros(len(tr)); p_te = np.zeros(len(te)); imp = pd.Series(0.0, index=feats)
    for f, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(tr, tr.target)):
        mdl = lgb.train(pc, lgb.Dataset(tr.loc[a, feats], tr.target[a]), 2000,
                        valid_sets=[lgb.Dataset(tr.loc[b, feats], tr.target[b])],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[b] = mdl.predict(tr.loc[b, feats]); p_te += mdl.predict(te[feats]) / 5
        imp += pd.Series(mdl.feature_importance("gain"), index=feats)
    print(f"Gini CV: {2 * roc_auc_score(tr.target, oof) - 1:.4f}")
    print("Top 15 variables (gain):\n", (imp / imp.sum()).sort_values(ascending=False).head(15).round(3).to_string())

    # monto: (a) regresion directa sobre log1p (optimiza el R2 de la metrica)
    pr = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_child_samples=50, verbose=-1, seed=7)
    r_all = lgb.train(pr, lgb.Dataset(tr[feats], np.log1p(tr.monto_target)), 600)
    # (b) monto condicional entre los que convierten -> para valor esperado de negocio
    pos = tr[tr.target == 1]
    r_pos = lgb.train(pr, lgb.Dataset(pos[feats], np.log1p(pos.monto_target)), 400)
    te["prob_factoring"] = p_te
    te["monto_esperado"] = np.expm1(r_all.predict(te[feats])).clip(0)
    te["monto_si_convierte"] = np.expm1(r_pos.predict(te[feats]))
    te["valor_esperado"] = te.prob_factoring * te.monto_si_convierte * 0.018 - 150
    return te


# ---------------------------------------------------------------- F
def asignar(te):
    K = int(round(0.025 * len(te)))
    te["priorizado"] = 0
    for z, p in CUPOS.items():
        c = int(K * p)
        idx = te[(te.zona == z) & (te.valor_esperado > 0)].nlargest(c, "valor_esperado").index
        te.loc[idx, "priorizado"] = 1
    print(f"Priorizados: {te.priorizado.sum()} de capacidad {K}")
    return te


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_lite")
    ap.add_argument("--salida", default="entrega_ref.csv")
    a = ap.parse_args()
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    limpieza_maestro(con, a.data)
    features_transacciones(con, a.data)
    features_factoring(con, a.data)
    features_perfil(con, a.data)
    te = asignar(modelar(con, a.data))
    te[["id_empresa", "prob_factoring", "monto_esperado", "priorizado"]].to_csv(a.salida, index=False)
    print(f"Entrega guardada en {a.salida}")
