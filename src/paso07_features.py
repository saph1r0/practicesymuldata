"""PASO 7 - Ingenieria de variables. Una tabla de features por fuente, todas con UNA fila por empresa (llave k).
Por tipo de tabla:
  - maestro (atributos fijos)        -> se usan casi tal cual
  - transacciones (eventos)          -> se AGREGAN por empresa: conteos, sumas, promedios, ventanas de tiempo, ratios
  - factoring (historial de eventos) -> igual, pero SOLO antes del corte (paso 5)
  - perfil (panel ancho 12 meses)    -> se RESUMEN: nivel reciente, tendencia, volatilidad"""
import numpy as np
import pandas as pd
from comun import pq, LIMPIO, FEAT, CORTE, conectar, cronometro

con = conectar()
L = lambda n: pq(LIMPIO, n)

# ------------------------------------------------------------- 1) MAESTRO
with cronometro("feat_maestro"):
    con.execute(f"""COPY (SELECT k, seg, sector, zona, cliente, lead, monto_oferta_factoring AS monto_oferta,
        ventas_anuales_soles AS ventas, num_trabajadores AS ntrab, antig_anios, activo, habido, tipo_contribuyente,
        rating_externo, score_buro, antiguedad_cliente_meses, num_establecimientos, flag_exportador,
        flag_importador, flag_agente_retencion, flag_buen_contribuyente, flag_grupo_economico
      FROM '{L("maestro")}') TO '{pq(FEAT, "f_maestro")}' (FORMAT parquet)""")

# ------------------------------------------------------------- 2) RED PAGADOR -> PROVEEDOR (transacciones)
with cronometro("feat_red"):
    con.execute(f"""
    CREATE TABLE fx AS                       -- solo pagos de facturas, con el segmento del PAGADOR
    SELECT x.*, p.seg AS seg_pag FROM '{L("transacciones")}' x LEFT JOIN '{L("maestro")}' p ON x.ord = p.k
    WHERE x.tipo IN ('PAGO_FACTURA', 'PAGO_PROVEEDORES_MASIVO');

    -- 38% de dias_credito es nulo: lo imputo con el plazo TIPICO de ese pagador (los pagadores pagan parecido)
    CREATE TABLE plazo_pagador AS SELECT ord, median(dias_credito) AS plazo_tipico FROM fx GROUP BY ord;

    COPY (
    SELECT f.ben AS k,
      count(*)                                   AS n_facturas,
      sum(monto_pen)                             AS vol_facturas,
      avg(monto_pen)                             AS ticket_prom,
      count(DISTINCT ord)                        AS n_pagadores,
      max(monto_pen) / sum(monto_pen)            AS max_share_factura,
      avg(coalesce(dias_credito, plazo_tipico))  AS plazo_prom,
      max(coalesce(dias_credito, plazo_tipico))  AS plazo_max,
      sum(monto_pen) FILTER (WHERE coalesce(dias_credito, plazo_tipico) >= 60) / sum(monto_pen) AS share_plazo60,
      sum(monto_pen) FILTER (WHERE seg_pag IN ('GRANDE','CORPORATIVO','ESTADO')) / sum(monto_pen) AS share_grandes,
      coalesce(sum(monto_pen) FILTER (WHERE seg_pag = 'ESTADO'), 0) AS vol_estado,
      -- ventanas de tiempo: lo reciente pesa mas que lo antiguo
      coalesce(sum(monto_pen) FILTER (WHERE fh >= DATE '{CORTE}' - INTERVAL 3 MONTH), 0) AS vol_ult3m,
      coalesce(sum(monto_pen) FILTER (WHERE fh >= DATE '{CORTE}' - INTERVAL 6 MONTH), 0) AS vol_ult6m,
      coalesce(sum(monto_pen) FILTER (WHERE fh <  DATE '{CORTE}' - INTERVAL 6 MONTH
                                        AND fh >= DATE '{CORTE}' - INTERVAL 12 MONTH), 0) AS vol_prev6m,
      count(DISTINCT date_trunc('month', fh))    AS meses_activos,
      date_diff('day', max(fh), DATE '{CORTE}')  AS dias_desde_ult_factura
    FROM fx f LEFT JOIN plazo_pagador USING (ord) GROUP BY f.ben
    ) TO '{pq(FEAT, "f_red")}' (FORMAT parquet);

    COPY (SELECT ord AS k, sum(monto_pen) AS vol_pagado_facturas, count(DISTINCT ben) AS n_proveedores
          FROM fx GROUP BY ord) TO '{pq(FEAT, "f_pagador")}' (FORMAT parquet);

    COPY (SELECT k, sum(n_in) AS n_transf_in, sum(n_out) AS n_transf_out FROM (
            SELECT ben AS k, count(*) n_in, 0 n_out FROM '{L("transacciones")}' WHERE tipo LIKE 'TRANSF%' GROUP BY 1
            UNION ALL SELECT ord, 0, count(*) FROM '{L("transacciones")}' WHERE tipo LIKE 'TRANSF%' GROUP BY 1) GROUP BY k
    ) TO '{pq(FEAT, "f_transf")}' (FORMAT parquet);
    """)

# ------------------------------------------------------------- 3) HISTORIAL DE FACTORING (solo antes del corte)
with cronometro("feat_factoring"):
    con.execute(f"""COPY (
    SELECT k,
      count(*) FILTER (WHERE banco = 'BCP')                  AS n_ops_bcp,
      count(*) FILTER (WHERE banco = 'OTRO')                 AS n_ops_otro,
      coalesce(sum(monto_pen) FILTER (WHERE banco = 'BCP'), 0)  AS monto_bcp_hist,
      coalesce(sum(monto_pen) FILTER (WHERE banco = 'OTRO'), 0) AS monto_otro_hist,
      date_diff('day', max(fecha_desembolso), DATE '{CORTE}') AS dias_desde_ult_op,
      avg(tasa_mensual)                                      AS tasa_prom_hist,
      avg(plazo_dias)                                        AS plazo_prom_fact,
      count(*) FILTER (WHERE estado = 'VENCIDO')             AS n_vencidas
    FROM '{L("factoring")}'
    WHERE fecha_desembolso <= '{CORTE}' AND estado <> 'ANULADO'      -- <- la linea que evita la fuga
    GROUP BY k) TO '{pq(FEAT, "f_factoring")}' (FORMAT parquet)""")

# ------------------------------------------------------------- 4) PERFIL: 12 meses -> resumen
with cronometro("feat_perfil"):
    p = pd.read_parquet(L("perfil"))
    out = p[["k", "calificacion_sbs", "num_productos_activos", "flag_prestamo_capital_trabajo", "flag_leasing",
             "flag_carta_fianza", "flag_telecredito", "score_riesgo_interno", "num_entidades_sistema",
             "linea_credito_aprobada"]].copy()
    meses = lambda v, a, b: [f"{v}_m{j:02d}" for j in range(a, b + 1)]
    for v in ["saldo_cta_cte", "uso_linea_pct", "deuda_sistema", "monto_abonos", "dias_atraso_max", "num_tx_abono"]:
        rec, ant = p[meses(v, 1, 3)].mean(axis=1), p[meses(v, 10, 12)].mean(axis=1)   # m01 = mas reciente
        out[f"{v}_rec3"] = rec                                    # nivel actual
        out[f"{v}_tend"] = (rec - ant) / (ant.abs() + 1)          # tendencia (+ sube, - baja)
        out[f"{v}_cv"] = p[meses(v, 1, 12)].std(axis=1) / (p[meses(v, 1, 12)].mean(axis=1).abs() + 1)  # volatilidad
    out["atraso_max12"] = p[meses("dias_atraso_max", 1, 12)].max(axis=1)
    out.to_parquet(pq(FEAT, "f_perfil"), index=False)

for f in ["f_maestro", "f_red", "f_pagador", "f_transf", "f_factoring", "f_perfil"]:
    n, c = con.execute(f"SELECT count(*), count(DISTINCT k) FROM '{pq(FEAT, f)}'").fetchone()
    cols = len(con.execute(f"DESCRIBE SELECT * FROM '{pq(FEAT, f)}'").fetchall()) - 1
    print(f"  {f:<12} filas={n:>8,}  k distintos={c:>8,}  variables={cols}")
