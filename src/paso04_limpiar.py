"""PASO 4 - Limpiar cada tabla UNA vez y guardarla en Parquet.
Regla: tablas grandes -> SQL en DuckDB (no cargan en RAM). Tablas chicas (<1M filas) -> pandas esta bien.
Cada regla imprime cuantas filas toca: si una regla elimina 40% de la data, algo esta mal."""
import unicodedata
import numpy as np
import pandas as pd
from comun import csv, pq, LIMPIO, TC, conectar, cronometro

con = conectar()

# ------------------------------------------------------------------ MAESTRO (400 mil filas -> pandas)
with cronometro("maestro"):
    m = con.execute(f"""
      SELECT * EXCLUDE rn FROM (
        SELECT *, k(id_empresa) AS k,
               row_number() OVER (PARTITION BY k(id_empresa) ORDER BY fecha_actualizacion DESC) AS rn
        FROM read_csv('{csv('maestro_empresas.csv')}', all_varchar=true)) WHERE rn = 1""").df()
    print(f"  dedup: 406,000 -> {len(m):,} filas (se queda el registro mas reciente)")

    def norm(s):  # mayusculas, sin espacios extremos, sin tildes
        s = s.fillna("").str.upper().str.strip()
        return s.map(lambda x: "".join(c for c in unicodedata.normalize("NFKD", x) if not unicodedata.combining(c)))

    seg = norm(m.segmento)
    m["seg"] = seg.replace({"MICRO": "MIPE", "MYPE": "MIPE", "PEQUENA EMPRESA": "PYME", "MEDIANA EMPRESA": "MEDIANA",
                            "GRAN EMPRESA": "GRANDE", "CORP": "CORPORATIVO", "GOBIERNO": "ESTADO", "SECTOR PUBLICO": "ESTADO"})
    print("  segmento:", seg.nunique(), "variantes ->", m.seg.nunique(), "valores:", sorted(m.seg.unique()))

    # rubro: el CIIU es mas confiable que el texto libre
    m["ciiu"] = m.ciiu_rev4.fillna("").str.replace(".0", "", regex=False).str.zfill(4).replace("0000", np.nan)
    mapa = (pd.DataFrame({"c": m.ciiu, "s": m.sector_economico_sbs}).dropna()
            .groupby("c").s.agg(lambda x: x.mode()[0]))           # sector mas frecuente de cada CIIU
    m["sector"] = m.ciiu.map(mapa).fillna(m.sector_economico_sbs).fillna("DESCONOCIDO")
    print(f"  sector: texto libre tenia {m.rubro_descripcion.nunique()} variantes -> {m.sector.nunique()} sectores;"
          f" desconocidos: {(m.sector == 'DESCONOCIDO').mean():.2%}")

    dep = norm(m.departamento).replace({"AQP": "AREQUIPA", "LIMA METROPOLITANA": "LIMA", "CUZCO": "CUSCO",
                                        "LALIBERTAD": "LA LIBERTAD", "PROV. CONST. DEL CALLAO": "CALLAO"})
    dep = dep.str.replace(r"[^A-Z ]", "", regex=True).str.replace("JUNAN", "JUNIN")   # encoding roto JUNÃN
    ZONA = {"LIMA": "LIMA", "CALLAO": "LIMA", "AREQUIPA": "SUR", "CUSCO": "SUR", "ICA": "SUR", "PUNO": "SUR", "TACNA": "SUR",
            "MOQUEGUA": "SUR", "LA LIBERTAD": "NORTE", "PIURA": "NORTE", "LAMBAYEQUE": "NORTE", "ANCASH": "NORTE",
            "CAJAMARCA": "NORTE", "JUNIN": "CENTRO", "HUANUCO": "CENTRO", "LORETO": "ORIENTE", "SAN MARTIN": "ORIENTE", "UCAYALI": "ORIENTE"}
    m["zona"] = dep.map(ZONA).fillna(m.zona_comercial).fillna("DESCONOCIDA")
    print(f"  zona: desde departamento {dep.map(ZONA).notna().mean():.1%}; desconocida final {(m.zona == 'DESCONOCIDA').mean():.2%}")

    m["cliente"] = norm(m.flag_cliente_bcp).isin(["S", "SI", "1", "Y"]).astype("int8")
    m["lead"] = norm(m.flag_lead_factoring).isin(["1", "S"]).astype("int8")
    for c, centinelas in [("ventas_anuales_soles", [-1, 0, 99999999]), ("num_trabajadores", [-999])]:
        x = pd.to_numeric(m[c], errors="coerce")
        print(f"  {c}: {x.isin(centinelas).sum():,} centinelas -> nulo")
        m[c] = x.where(~x.isin(centinelas))
    fc = pd.to_datetime(m.fecha_constitucion, errors="coerce", format="mixed", dayfirst=True)
    print(f"  fecha_constitucion: {(fc.dt.year == 1900).sum():,} con 1900-01-01 -> nulo; no parseables: {fc.isna().sum():,}")
    m["antig_anios"] = ((pd.Timestamp("2025-06-30") - fc.where(fc.dt.year > 1900)).dt.days / 365).round(1)
    num = ["score_buro", "antiguedad_cliente_meses", "num_establecimientos", "monto_oferta_factoring", "flag_exportador",
           "flag_importador", "flag_agente_retencion", "flag_buen_contribuyente", "flag_grupo_economico"]
    m[num] = m[num].apply(pd.to_numeric, errors="coerce")
    m["activo"] = (m.estado_contribuyente == "ACTIVO").astype("int8")
    m["habido"] = (m.condicion_domicilio == "HABIDO").astype("int8")
    # columnas que NO pasan: identificadores (hashes, codigo_interno), constantes (pais), texto ya normalizado
    keep = ["k", "id_empresa", "seg", "sector", "ciiu", "zona", "cliente", "lead", "ventas_anuales_soles", "num_trabajadores",
            "antig_anios", "activo", "habido", "tipo_contribuyente", "rating_externo", "banca"] + num
    m[keep].to_parquet(pq(LIMPIO, "maestro"), index=False)
    print(f"  -> limpio/maestro.parquet {m[keep].shape}")

# ------------------------------------------------------------------ TRANSACCIONES (7.9 M filas -> DuckDB)
with cronometro("transacciones"):
    con.execute(f"CREATE VIEW tx_raw AS SELECT * FROM read_csv('{csv('transacciones.csv')}', all_varchar=true)")
    con.execute(f"""
    CREATE TABLE tx AS
    SELECT DISTINCT * FROM tx_raw;                                  -- 1) copias exactas fuera
    """)
    n0 = con.execute("SELECT count(*) FROM tx_raw").fetchone()[0]
    n1 = con.execute("SELECT count(*) FROM tx").fetchone()[0]
    con.execute(f"""
    CREATE TABLE txl AS
    SELECT k(id_ordenante) AS ord, k(id_beneficiario) AS ben,
      COALESCE(try_strptime(fecha_hora, '%Y-%m-%d %H:%M:%S'),
               try_strptime(fecha_hora, '%d/%m/%Y %H:%M')) AS fh,     -- 2) dos formatos de fecha
      tipo_transaccion AS tipo,
      TRY_CAST(replace(monto, ',', '') AS DOUBLE)                     -- 3) "1,234.50" -> 1234.50
        * CASE WHEN upper(trim(moneda)) IN ('USD', 'US$') THEN {TC} ELSE 1 END AS monto_pen,   -- 4) todo a soles
      TRY_CAST(dias_credito_documento AS DOUBLE) AS dias_credito,
      estado_tx
    FROM tx""")
    n2 = con.execute("SELECT count(*) FROM txl WHERE estado_tx = 'PROCESADA'").fetchone()[0]
    malos = con.execute("SELECT count(*) FILTER (WHERE fh IS NULL), count(*) FILTER (WHERE monto_pen IS NULL), "
                        "count(*) FILTER (WHERE monto_pen > 5e8), count(*) FILTER (WHERE monto_pen <= 0) FROM txl").fetchone()
    con.execute(f"""COPY (SELECT ord, ben, fh, tipo, monto_pen, dias_credito FROM txl
                         WHERE estado_tx = 'PROCESADA' AND fh IS NOT NULL
                           AND monto_pen > 0 AND monto_pen <= 5e8)          -- 5) sin extornos, ni centinelas
                    TO '{pq(LIMPIO, 'transacciones')}' (FORMAT parquet)""")
    n3 = con.execute(f"SELECT count(*) FROM '{pq(LIMPIO, 'transacciones')}'").fetchone()[0]
    print(f"  crudo {n0:,} -> sin duplicados {n1:,} -> PROCESADA {n2:,} -> final {n3:,}")
    print(f"  fechas no parseadas {malos[0]:,} | montos no numericos {malos[1]:,} | > 500 M {malos[2]:,} | <= 0 {malos[3]:,}")
    con.execute("DROP TABLE tx; DROP TABLE txl")

# ------------------------------------------------------------------ FACTORING (220 mil filas -> DuckDB)
with cronometro("factoring"):
    con.execute(f"""COPY (
      SELECT id_operacion, k(id_proveedor) AS k, k(id_pagador) AS k_pagador,
        CASE WHEN entidad_financiera IN ('BCP', '002') THEN 'BCP' ELSE 'OTRO' END AS banco,
        fuente_registro, estado_operacion AS estado, CAST(fecha_desembolso AS DATE) AS fecha_desembolso,
        TRY_CAST(plazo_dias AS DOUBLE) AS plazo_dias,
        TRY_CAST(monto_facturas AS DOUBLE) * CASE WHEN moneda = 'USD' THEN {TC} ELSE 1 END AS monto_pen,
        TRY_CAST(replace(tasa_mensual, '%', '') AS DOUBLE) AS tasa_mensual, segmento_pagador
      FROM read_csv('{csv('factoring_operaciones.csv')}', all_varchar=true)
      WHERE NOT (fuente_registro = 'SPI' AND entidad_financiera = '002')   -- misma operacion BCP vista por el SPI
    ) TO '{pq(LIMPIO, 'factoring')}' (FORMAT parquet)""")
    print(con.execute(f"SELECT banco, fuente_registro, count(*) n FROM '{pq(LIMPIO, 'factoring')}' GROUP BY ALL ORDER BY 1, 2").df().to_string(index=False))

# ------------------------------------------------------------------ PERFIL (340 mil x 116 -> pandas)
with cronometro("perfil"):
    p = con.execute(f"SELECT * FROM read_csv('{csv('perfil_empresa.csv')}', all_varchar=true)").df()
    p.insert(0, "k", pd.to_numeric(p.id_empresa.str.replace("EMP", ""), errors="coerce"))
    fugas = ["flag_factoring_activo", "fecha_ultimo_contacto_comercial", "resultado_ultimo_contacto"]
    p = p.drop(columns=fugas + ["periodo_corte", "id_empresa"])        # ver paso 5: informacion del futuro
    numcols = [c for c in p.columns if c not in ("k", "calificacion_sbs")]
    p[numcols] = p[numcols].apply(pd.to_numeric, errors="coerce").astype("float32")   # float32: la mitad de memoria
    print(f"  -999 -> nulo: {(p[numcols] == -999).sum().sum():,} celdas")
    p[numcols] = p[numcols].replace(-999, np.nan)
    p.to_parquet(pq(LIMPIO, "perfil"), index=False)
    print(f"  -> limpio/perfil.parquet {p.shape} | eliminadas por fuga: {fugas}")
