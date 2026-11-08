"""PASO 6 - EDA guiado por hipotesis (solo con TRAIN; mirar test seria hacer trampa).
Cada bloque responde una hipotesis de negocio con una tabla de tasas: eso es lo que luego va a las laminas."""
import pandas as pd
from comun import pq, LIMPIO, FEAT, conectar

con = conectar()
con.execute(f"CREATE VIEW m AS SELECT * FROM '{pq(LIMPIO, 'maestro')}'")
con.execute(f"CREATE VIEW t AS SELECT * FROM '{pq(FEAT, 'target')}' WHERE es_test = 0")
# por proveedor: volumen facturado a grandes pagadores y plazo promedio (version rapida; el paso 7 la refina)
con.execute(f"""
CREATE TABLE red AS
SELECT x.ben AS k, sum(x.monto_pen) AS vol, avg(x.dias_credito) AS plazo,
       sum(x.monto_pen) FILTER (WHERE p.seg IN ('GRANDE','CORPORATIVO','ESTADO')) / sum(x.monto_pen) AS share_grandes,
       max((p.seg = 'ESTADO')::INT) AS vende_estado
FROM '{pq(LIMPIO, 'transacciones')}' x JOIN m p ON x.ord = p.k
WHERE x.tipo IN ('PAGO_FACTURA', 'PAGO_PROVEEDORES_MASIVO') GROUP BY 1""")
pd.set_option("display.width", 160)


def tabla(titulo, sql):
    print(f"\n{titulo}\n" + con.execute(sql).df().to_string(index=False))


tabla("H2 - Tasa de conversion por rubro (proveedores, sin Estado ni corporativos)", """
  SELECT sector, count(*) empresas, round(100*avg(target), 1) AS tasa_pct, round(avg(monto_target) FILTER (WHERE target=1)/1e3) AS monto_medio_miles
  FROM t JOIN m USING (k) WHERE seg NOT IN ('ESTADO','CORPORATIVO') GROUP BY 1 ORDER BY tasa_pct DESC""")

tabla("H1 - Tasa segun plazo de cobro de sus facturas", """
  SELECT CASE WHEN r.k IS NULL THEN '0 sin facturas B2B' WHEN plazo < 45 THEN '1 <45 dias' WHEN plazo < 60 THEN '2 45-59'
              WHEN plazo < 90 THEN '3 60-89' ELSE '4 90+' END AS plazo_cobro,
         count(*) empresas, round(100*avg(target), 1) tasa_pct
  FROM t JOIN m USING (k) LEFT JOIN red r USING (k) WHERE m.seg NOT IN ('ESTADO','CORPORATIVO') GROUP BY 1 ORDER BY 1""")

tabla("H1b - Construccion: le vende al Estado o no", """
  SELECT vende_estado, count(*) empresas, round(100*avg(target), 1) tasa_pct
  FROM t JOIN m USING (k) JOIN red r USING (k) WHERE m.sector = 'CONSTRUCCION' GROUP BY 1 ORDER BY 1""")

tabla("H6 - Lead actual vs Pyme sin lead, por plazo de cobro (la regla del lead ignora el plazo?)", """
  SELECT lead, CASE WHEN plazo >= 60 THEN 'plazo >= 60' ELSE 'plazo < 60 o sin dato' END AS plazo,
         count(*) empresas, round(100*avg(target), 1) tasa_pct
  FROM t JOIN m USING (k) LEFT JOIN red r USING (k)
  WHERE m.seg IN ('MIPE','PYME') AND m.cliente = 1 GROUP BY 1, 2 ORDER BY 1, 2""")

tabla("Tamano del mercado: proveedores B2B por rubro y cuanto les facturan grandes pagadores (S/ millones)", """
  SELECT sector, count(*) proveedores, round(sum(vol*share_grandes)/1e6) AS vol_grandes_MM,
         round(100*avg((plazo >= 60)::INT), 1) AS pct_plazo_60
  FROM red JOIN m USING (k) WHERE m.seg NOT IN ('ESTADO','CORPORATIVO') GROUP BY 1 ORDER BY 3 DESC LIMIT 8""")
