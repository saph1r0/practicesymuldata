#!/usr/bin/env python3
"""
SIMULACRO DATAFEST BCP - Caso Factoring por Rubros
===================================================
Genera 4 tablas CRUDAS (sin preprocesar, con errores reales de calidad) que imitan
lo anunciado en la reunion del 23/10:
  1. maestro_empresas.csv      (~45 columnas)  clientes y no clientes, rubro, tamano, ubicacion
  2. transacciones.csv         (~22 columnas)  pagos de facturas, transferencias -> red pagador/proveedor
  3. factoring_operaciones.csv (~28 columnas)  historial de factoring (BCP interno + SPI otros bancos)
  4. perfil_empresa.csv        (~116 columnas) productos y evolucion de saldos 12 meses (formato ancho)
Ademas:
  ids_evaluacion.csv        -> empresas sobre las que deben entregar prediccion (set ciego)
  _clave_oculta/clave.csv   -> respuestas (NO ABRIR durante el simulacro; la usa evaluar.py)

Uso:
  python generar_datafest_sim.py --escala 1.0 --salida data_full   # ~8M filas, ~210 columnas (~1.5 GB)
  python generar_datafest_sim.py --escala 0.1 --salida data_lite   # ~800k filas (para practicar rapido)

La escala 1.0 necesita ~6 GB de RAM libre y tarda 10-20 min.
"""
import argparse
import os
import numpy as np
import pandas as pd

TC = 3.75  # tipo de cambio referencial (se comunica en el enunciado)
INI = np.datetime64("2024-01-01")
FIN_OBS = np.datetime64("2025-06-30")      # fin de ventana de observacion
INI_TGT = np.datetime64("2025-07-01")      # ventana target
FIN_TGT = np.datetime64("2025-12-31")

# --------------------------------------------------------------------------------------
# Catalogos de negocio
# --------------------------------------------------------------------------------------
# sector: (peso, prob_b2b, efecto_propension, [ciiu], meses_pico)
SECTORES = {
    "AGROEXPORTACION":        (0.05, 0.85, 0.45, ["0111", "0121", "0122", "1030"], [9, 10, 11, 12]),
    "AGROPECUARIO":           (0.06, 0.50, 0.10, ["0111", "0141", "0145"], [3, 4, 5]),
    "PESCA":                  (0.02, 0.70, 0.25, ["0311", "1020"], [4, 5, 11, 12]),
    "MINERIA":                (0.02, 0.85, 0.00, ["0710", "0729", "0990"], []),
    "MANUFACTURA":            (0.12, 0.80, 0.55, ["1010", "1071", "1410", "2220", "2511"], []),
    "CONSTRUCCION":           (0.09, 0.85, 0.60, ["4100", "4210", "4290", "4321"], [11, 12]),
    "COMERCIO_MAYORISTA":     (0.14, 0.75, 0.35, ["4630", "4649", "4659", "4690"], [11, 12]),
    "COMERCIO_MINORISTA":     (0.17, 0.15, -0.80, ["4711", "4719", "4771", "4730"], [12]),
    "TRANSPORTE_LOGISTICA":   (0.07, 0.80, 0.45, ["4923", "5210", "5229"], []),
    "SERVICIOS_EMPRESARIALES": (0.09, 0.80, 0.30, ["6920", "7020", "7110", "7730", "8110", "8121"], []),
    "TECNOLOGIA":             (0.03, 0.85, 0.15, ["6201", "6202", "6311"], []),
    "SALUD":                  (0.03, 0.50, 0.00, ["8610", "8620", "2100"], []),
    "EDUCACION":              (0.03, 0.10, -0.60, ["8510", "8521", "8530"], [3]),
    "HOTELES_RESTAURANTES":   (0.05, 0.25, -0.50, ["5510", "5610", "5629"], [7, 12]),
    "ENERGIA":                (0.03, 0.80, 0.15, ["3510", "3520"], []),
}
SEC_NOMBRES = list(SECTORES.keys())

# variantes sucias del texto de rubro
SEC_VARIANTES = {
    "AGROEXPORTACION": ["Agroexportacion", "AGRO-EXPORTACION", "agroexportación", "AGROEXP."],
    "AGROPECUARIO": ["Agropecuario", "AGRICULTURA", "agro"],
    "PESCA": ["Pesca", "PESQUERO", "pesca "],
    "MINERIA": ["Mineria", "MINERÍA", "MINERIA E HIDROCARB."],
    "MANUFACTURA": ["Manufactura", "manufactura ", "MANUFAC.", "INDUSTRIA MANUFACTURERA"],
    "CONSTRUCCION": ["Construccion", "CONSTRUCCIÓN", "construcción", "CONST."],
    "COMERCIO_MAYORISTA": ["Comercio Mayorista", "COMERCIO AL POR MAYOR", "COM. MAYORISTA"],
    "COMERCIO_MINORISTA": ["Comercio Minorista", "COMERCIO AL POR MENOR", "RETAIL", "COM. MINORISTA"],
    "TRANSPORTE_LOGISTICA": ["Transporte", "TRANSPORTE Y ALMACENAMIENTO", "LOGISTICA"],
    "SERVICIOS_EMPRESARIALES": ["Servicios", "SERV. EMPRESARIALES", "servicios prestados a empresas"],
    "TECNOLOGIA": ["Tecnologia", "TI", "TECNOLOGÍA DE LA INFORMACIÓN"],
    "SALUD": ["Salud", "SERVICIOS DE SALUD"],
    "EDUCACION": ["Educacion", "EDUCACIÓN", "ENSEÑANZA"],
    "HOTELES_RESTAURANTES": ["Hoteles y Restaurantes", "HORECA", "ALOJAMIENTO Y COMIDAS"],
    "ENERGIA": ["Energia", "ELECTRICIDAD Y GAS", "ENERGÍA"],
    "ADMINISTRACION_PUBLICA": ["Adm. Publica", "SECTOR PUBLICO", "ADMINISTRACIÓN PÚBLICA"],
}

SEGMENTOS = ["MIPE", "PYME", "MEDIANA", "GRANDE", "CORPORATIVO", "ESTADO"]
SEG_P = [0.46, 0.30, 0.12, 0.07, 0.03, 0.02]
SEG_VARIANTES = {
    "MIPE": ["Mipe", "MICRO", "mype", "MYPE"],
    "PYME": ["Pyme", "PEQUEÑA EMPRESA", "pyme "],
    "MEDIANA": ["Mediana", "MEDIANA EMPRESA"],
    "GRANDE": ["Grande", "GRAN EMPRESA"],
    "CORPORATIVO": ["Corporativo", "CORP"],
    "ESTADO": ["Estado", "GOBIERNO", "SECTOR PUBLICO"],
}

DEPTOS = {  # depto: (peso, zona)
    "LIMA": (0.46, "LIMA"), "CALLAO": (0.04, "LIMA"), "AREQUIPA": (0.07, "SUR"),
    "LA LIBERTAD": (0.06, "NORTE"), "PIURA": (0.05, "NORTE"), "LAMBAYEQUE": (0.04, "NORTE"),
    "CUSCO": (0.04, "SUR"), "JUNIN": (0.04, "CENTRO"), "ICA": (0.04, "SUR"),
    "ANCASH": (0.03, "NORTE"), "CAJAMARCA": (0.02, "NORTE"), "PUNO": (0.02, "SUR"),
    "LORETO": (0.02, "ORIENTE"), "SAN MARTIN": (0.02, "ORIENTE"), "TACNA": (0.015, "SUR"),
    "UCAYALI": (0.015, "ORIENTE"), "HUANUCO": (0.01, "CENTRO"), "MOQUEGUA": (0.01, "SUR"),
}
DEP_NOM = list(DEPTOS.keys())
DEP_P = np.array([v[0] for v in DEPTOS.values()]); DEP_P = DEP_P / DEP_P.sum()
DEP_VARIANTES = {
    "LIMA": ["Lima", "LIMA ", "LIMA METROPOLITANA", "lima"], "AREQUIPA": ["Arequipa", "AQP", "AREQUIPA "],
    "CUSCO": ["CUZCO", "Cusco"], "JUNIN": ["JUNÍN", "JUNÃ\x8dN", "Junin"], "LA LIBERTAD": ["LA LIBERTAD ", "La Libertad", "LALIBERTAD"],
    "SAN MARTIN": ["SAN MARTÍN", "San Martin"], "HUANUCO": ["HUÁNUCO", "HUANUCO "], "CALLAO": ["Callao", "PROV. CONST. DEL CALLAO"],
    "PIURA": ["Piura"], "ANCASH": ["ÁNCASH", "Ancash"],
}
BANCOS_COD = {"BCP": "002", "INTERBANK": "003", "SCOTIABANK": "009", "BBVA": "011", "BANBIF": "038", "PICHINCHA": "035", "CAJA_AREQUIPA": "801"}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ensuciar_texto(rng, limpio, variantes, p):
    """Reemplaza una fraccion p de valores por variantes sucias."""
    out = np.array(limpio, dtype=object)
    m = rng.random(len(out)) < p
    idx = np.where(m)[0]
    for i in idx:
        v = variantes.get(out[i])
        if v:
            out[i] = v[rng.integers(len(v))]
    return out


def fechas_mixtas(rng, fechas, p_alt=0.12, p_sent=0.01):
    """Convierte datetime64 a texto con formatos mezclados y centinelas."""
    s = pd.to_datetime(pd.Series(fechas))
    iso = s.dt.strftime("%Y-%m-%d").to_numpy(dtype=object)
    r = rng.random(len(iso))
    alt1 = s.dt.strftime("%d/%m/%Y").to_numpy(dtype=object)
    alt2 = s.dt.strftime("%Y%m%d").to_numpy(dtype=object)
    iso = np.where(r < p_alt / 2, alt1, iso)
    iso = np.where((r >= p_alt / 2) & (r < p_alt), alt2, iso)
    iso = np.where((r >= p_alt) & (r < p_alt + p_sent), "1900-01-01", iso)
    return iso


def hex_ids(rng, n, largo=16):
    a = rng.integers(0, 16, size=(n, largo))
    alf = np.array(list("0123456789abcdef"))
    return np.array(["".join(x) for x in alf[a]], dtype=object)


def emp_id(i):
    return np.char.add("EMP", np.char.zfill(np.asarray(i).astype(str), 8))


# --------------------------------------------------------------------------------------
def generar(escala, salida, seed):
    rng = np.random.default_rng(seed)
    os.makedirs(salida, exist_ok=True)
    os.makedirs(os.path.join(salida, "_clave_oculta"), exist_ok=True)

    N = int(400_000 * escala)
    TX_OBJ = int(7_000_000 * escala)
    print(f"[1/6] Empresas: {N:,}")

    # ---------------- atributos latentes de empresa ----------------
    ids = np.arange(1, N + 1) * 7 + 1000          # ids no consecutivos
    seg = rng.choice(len(SEGMENTOS), size=N, p=SEG_P)
    segn = np.array(SEGMENTOS)[seg]
    sec_p = np.array([v[0] for v in SECTORES.values()]); sec_p /= sec_p.sum()
    sec = rng.choice(len(SEC_NOMBRES), size=N, p=sec_p)
    secn = np.array(SEC_NOMBRES, dtype=object)[sec]
    secn[segn == "ESTADO"] = "ADMINISTRACION_PUBLICA"

    # ubicacion (correlacionada con rubro)
    dep = rng.choice(len(DEP_NOM), size=N, p=DEP_P)
    depn = np.array(DEP_NOM, dtype=object)[dep]
    m_agro = secn == "AGROEXPORTACION"
    depn[m_agro] = rng.choice(["ICA", "PIURA", "LA LIBERTAD", "LAMBAYEQUE", "LIMA"], size=m_agro.sum(), p=[.3, .25, .25, .1, .1])
    m_min = secn == "MINERIA"
    depn[m_min] = rng.choice(["AREQUIPA", "CAJAMARCA", "ANCASH", "MOQUEGUA", "LIMA"], size=m_min.sum(), p=[.25, .2, .2, .15, .2])
    zona = np.array([DEPTOS[d][1] for d in depn], dtype=object)

    cliente_p = pd.Series(segn).map({"MIPE": .33, "PYME": .48, "MEDIANA": .60, "GRANDE": .70, "CORPORATIVO": .80, "ESTADO": .50}).to_numpy()
    cliente_p = cliente_p + 0.10 * (zona == "LIMA")
    cliente = rng.random(N) < cliente_p
    activo = rng.random(N) < 0.93
    exportador = (rng.random(N) < np.where(m_agro, 0.7, 0.06))
    ciiu = np.array([SECTORES[s][3][rng.integers(len(SECTORES[s][3]))] if s in SECTORES else ["8411", "8412", "8421"][rng.integers(3)] for s in secn], dtype=object)

    # ---------------- red pagador -> proveedor ----------------
    print("[2/6] Construyendo red pagador-proveedor")
    es_pagador = np.isin(segn, ["GRANDE", "CORPORATIVO", "ESTADO"]) | ((segn == "MEDIANA") & (rng.random(N) < 0.2))
    pag_idx = np.where(es_pagador)[0]
    pag_plazo = pd.Series(segn[pag_idx]).map({"ESTADO": 95, "CORPORATIVO": 72, "GRANDE": 60, "MEDIANA": 40}).to_numpy().astype(float)
    pag_plazo += np.where(secn[pag_idx] == "MINERIA", 15, 0) + np.where(secn[pag_idx] == "COMERCIO_MINORISTA", 22, 0)
    pag_plazo += rng.normal(0, 10, len(pag_idx))
    pag_plazo = np.clip(pag_plazo, 15, 150)
    pag_peso = pd.Series(segn[pag_idx]).map({"ESTADO": 4, "CORPORATIVO": 6, "GRANDE": 2.5, "MEDIANA": 1}).to_numpy() * rng.lognormal(0, 0.6, len(pag_idx))
    plazo_de = np.zeros(N); plazo_de[pag_idx] = pag_plazo

    es_prov = np.isin(segn, ["MIPE", "PYME", "MEDIANA", "GRANDE"])
    b2b_p = np.array([SECTORES[s][1] if s in SECTORES else 0 for s in secn])
    seg_f = pd.Series(segn).map({"MIPE": .55, "PYME": 1.0, "MEDIANA": 1.1, "GRANDE": .9, "CORPORATIVO": 0, "ESTADO": 0}).to_numpy()
    b2b = es_prov & (rng.random(N) < np.clip(b2b_p * seg_f, 0, 0.95))
    k_mean = pd.Series(segn).map({"MIPE": .4, "PYME": 1.0, "MEDIANA": 2.2, "GRANDE": 3.0, "CORPORATIVO": 0, "ESTADO": 0}).to_numpy()
    k = np.where(b2b, 1 + rng.poisson(k_mean), 0)
    src = np.repeat(np.arange(N), k)            # proveedor
    E = len(src)
    # eleccion de pagador con afinidad sectorial
    grp_estado = np.where(segn[pag_idx] == "ESTADO")[0]
    grp_min = np.where(secn[pag_idx] == "MINERIA")[0]
    grp_ret = np.where(secn[pag_idx] == "COMERCIO_MINORISTA")[0]

    def pick(grupo, n):
        w = pag_peso[grupo]; w = w / w.sum()
        return grupo[rng.choice(len(grupo), size=n, p=w)]
    todos = np.arange(len(pag_idx))
    dst_local = pick(todos, E)
    s_src = secn[src]
    r = rng.random(E)
    m = (s_src == "CONSTRUCCION") & (r < 0.45) & (len(grp_estado) > 0)
    dst_local[m] = pick(grp_estado, m.sum())
    m = np.isin(s_src, ["SERVICIOS_EMPRESARIALES", "TRANSPORTE_LOGISTICA", "ENERGIA"]) & (r < 0.25) & (len(grp_min) > 0)
    dst_local[m] = pick(grp_min, m.sum())
    m = np.isin(s_src, ["AGROPECUARIO", "MANUFACTURA", "COMERCIO_MAYORISTA"]) & (r < 0.30) & (len(grp_ret) > 0)
    dst_local[m] = pick(grp_ret, m.sum())
    dst = pag_idx[dst_local]
    keep = dst != src
    src, dst = src[keep], dst[keep]
    E = len(src)
    base_amt = pd.Series(segn[src]).map({"MIPE": 6_000, "PYME": 25_000, "MEDIANA": 110_000, "GRANDE": 450_000}).to_numpy()
    amt = base_amt * rng.lognormal(0, 0.7, E) * np.where(segn[dst] == "CORPORATIVO", 1.5, 1.0)
    freq = rng.gamma(2.0, 0.45, E)                         # facturas/mes
    eplazo = np.clip(plazo_de[dst] + rng.normal(0, 8, E), 7, 180)

    # agregados por proveedor (latentes)
    vol = np.bincount(src, weights=amt * freq * 18, minlength=N)
    plz = np.bincount(src, weights=eplazo * amt * freq, minlength=N) / np.maximum(np.bincount(src, weights=amt * freq, minlength=N), 1e-9)
    has_estado = np.bincount(src, weights=(segn[dst] == "ESTADO").astype(float), minlength=N) > 0
    mean_inv = np.bincount(src, weights=amt, minlength=N) / np.maximum(np.bincount(src, minlength=N), 1)
    mean_inv = np.where(mean_inv > 0, mean_inv, pd.Series(segn).map({"MIPE": 4000, "PYME": 15000, "MEDIANA": 70000, "GRANDE": 300000, "CORPORATIVO": 1e6, "ESTADO": 1e6}).to_numpy())

    stress = rng.normal(0, 1, N) + 0.35 * (plz >= 75)

    # ---------------- propension a factoring ----------------
    sec_eff = np.array([SECTORES[s][2] if s in SECTORES else -5 for s in secn])
    seg_eff = pd.Series(segn).map({"MIPE": -0.5, "PYME": 0.35, "MEDIANA": 0.5, "GRANDE": -0.3, "CORPORATIVO": -3.0, "ESTADO": -8}).to_numpy()
    z = (-5.4 + sec_eff + seg_eff + 0.42 * np.log1p(vol / 1e4) + 0.8 * (plz >= 60) + 0.6 * (plz >= 90)
         + 0.45 * stress + 0.6 * (has_estado & (secn == "CONSTRUCCION")) + 0.3 * exportador
         - 1.2 * (~activo) + rng.normal(0, 1.4, N))
    hist = rng.random(N) < sigmoid(z)
    hist_bcp = hist & cliente & (rng.random(N) < 0.62)
    hist_otro = hist & ~hist_bcp
    cliente = cliente | hist_bcp
    # ---------------- LEADS (grupo A): ~10% de clientes Pyme con oferta precalificada ----------------
    # El banco los eligio con una REGLA ANTIGUA: tamano, historial BCP y volumen; NO mira plazo, rubro ni estres.
    # Por eso hay empresas con alto potencial fuera de la lista (grupo B: "habilitar mas").
    pyme_cli = cliente & np.isin(segn, ["MIPE", "PYME"])
    regla = 1.5 * hist_bcp + 0.35 * np.log1p(vol / 1e4) + 0.4 * (segn == "PYME") + rng.normal(0, 1.0, N)
    corte = np.quantile(regla[pyme_cli], 0.90)
    lead = pyme_cli & (regla >= corte)
    zt = (z + 1.7 * hist_bcp + 0.6 * hist_otro + 0.8 * cliente + 1.3 * lead + 0.35 * (zona == "LIMA")
          + 0.5 * m_agro - 2.5 + rng.normal(0, 1.5, N))
    target = (rng.random(N) < sigmoid(zt)) & es_prov
    n_ops_t = np.where(target, 1 + rng.poisson(0.8 + 2.0 * hist_bcp), 0)
    monto_target = n_ops_t * mean_inv * rng.uniform(1, 4, N) * np.exp(0.15 * stress)
    monto_target = np.where(target, monto_target, 0.0)
    print(f"      proveedores B2B: {b2b.mean():.1%} | historico factoring: {hist[es_prov].mean():.1%} | "
          f"target (elegibles): {target[es_prov].mean():.2%}")

    # split train / test
    test = rng.random(N) < 0.30

    # ======================================================================
    # TABLA 1: maestro_empresas
    # ======================================================================
    print("[3/6] maestro_empresas")
    fconst = INI - rng.integers(200, 365 * 35, N).astype("timedelta64[D]")
    ntrab = np.round(pd.Series(segn).map({"MIPE": 4, "PYME": 25, "MEDIANA": 120, "GRANDE": 600, "CORPORATIVO": 3000, "ESTADO": 1500}).to_numpy() * rng.lognormal(0, 0.6, N)).astype(int)
    ventas = pd.Series(segn).map({"MIPE": 2e5, "PYME": 2.5e6, "MEDIANA": 2e7, "GRANDE": 1.5e8, "CORPORATIVO": 1.2e9, "ESTADO": 5e8}).to_numpy() * rng.lognormal(0, 0.7, N)
    ventas = ventas.round(0).astype(object)
    rv = rng.random(N)
    ventas[rv < 0.07] = np.nan
    ventas[(rv >= 0.07) & (rv < 0.09)] = -1
    ventas[(rv >= 0.09) & (rv < 0.10)] = 99999999
    flag_cli = np.where(cliente, rng.choice(["S", "SI", "1", "Y"], N, p=[.6, .2, .15, .05]), rng.choice(["N", "NO", "0", ""], N, p=[.6, .15, .15, .1])).astype(object)
    flag_cli[rng.random(N) < 0.03] = np.nan
    alta = np.where(cliente, (INI + rng.integers(-3650, 500, N).astype("timedelta64[D]")).astype("datetime64[D]").astype(str), "")
    banca = np.where(~cliente, "NO_CLIENTE", pd.Series(segn).map({"MIPE": "BANCA_PEQ_EMPRESA", "PYME": "BANCA_PEQ_EMPRESA", "MEDIANA": "BANCA_EMPRESA", "GRANDE": "BANCA_EMPRESA", "CORPORATIVO": "BANCA_CORPORATIVA", "ESTADO": "BANCA_INSTITUCIONAL"}).to_numpy())
    sec_txt = ensuciar_texto(rng, secn, SEC_VARIANTES, 0.25)
    sec_txt[rng.random(N) < 0.05] = np.nan
    sec_txt[rng.random(N) < 0.02] = "SIN INFO"
    ciiu_txt = ciiu.copy()
    rc = rng.random(N)
    ciiu_txt[rc < 0.15] = [c.lstrip("0") for c in ciiu_txt[rc < 0.15]]
    ciiu_txt[(rc >= 0.15) & (rc < 0.22)] = [c + ".0" for c in ciiu_txt[(rc >= 0.15) & (rc < 0.22)]]
    ciiu_txt[(rc >= 0.22) & (rc < 0.25)] = np.nan
    dep_txt = ensuciar_texto(rng, depn, DEP_VARIANTES, 0.18)
    dep_txt[rng.random(N) < 0.02] = np.nan
    zona_txt = zona.copy(); zona_txt[rng.random(N) < 0.12] = np.nan
    seg_txt = ensuciar_texto(rng, segn, SEG_VARIANTES, 0.15)
    rating = rng.choice(["A", "B", "C", "D", "E", "NR"], N, p=[.15, .3, .25, .1, .05, .15])
    score_buro = np.clip(rng.normal(600 - 40 * stress, 80), 150, 999).round(0).astype(object)
    score_buro[rng.random(N) < 0.1] = np.nan
    tamano = np.where(ventas == -1, "NO DETERMINADO", pd.Series(segn).map({"MIPE": "MICRO", "PYME": "PEQUEÑA", "MEDIANA": "MEDIANA", "GRANDE": "GRANDE", "CORPORATIVO": "GRANDE", "ESTADO": "GRANDE"}).to_numpy())
    rango_trab = pd.cut(ntrab, [-1, 10, 50, 200, 1000, 1e9], labels=["1-10", "11-50", "51-200", "201-1000", "1000+"]).astype(str)
    grupo = np.where(rng.random(N) < np.where(np.isin(segn, ["GRANDE", "CORPORATIVO"]), 0.5, 0.04), "GE" + rng.integers(1, max(2, N // 200), N).astype(str), "")
    f_act = (INI + rng.integers(0, 547, N).astype("timedelta64[D]")).astype(str)

    maestro = pd.DataFrame({
        "id_empresa": emp_id(ids),
        "codigo_interno": np.arange(N) + 500000,
        "ruc_hash": hex_ids(rng, N),
        "razon_social_hash": hex_ids(rng, N, 20),
        "nombre_comercial_hash": np.where(rng.random(N) < 0.4, "", hex_ids(rng, N, 12)),
        "tipo_contribuyente": np.where(segn == "ESTADO", "ENTIDAD_PUBLICA", rng.choice(["SAC", "SA", "EIRL", "SRL", "PERSONA_NATURAL_NEG"], N, p=[.45, .15, .2, .1, .1])),
        "ciiu_rev4": ciiu_txt,
        "rubro_descripcion": sec_txt,
        "sector_economico_sbs": np.where(rng.random(N) < 0.3, np.nan, secn),
        "subsector": [f"SUB_{c}" for c in ciiu],
        "segmento": seg_txt,
        "tamano_empresa": tamano,
        "num_trabajadores": np.where(rng.random(N) < 0.08, -999, ntrab),
        "rango_trabajadores": rango_trab,
        "ventas_anuales_soles": ventas,
        "rango_ventas": pd.cut(pd.to_numeric(pd.Series(ventas), errors="coerce"), [-10, 0, 1.7e6, 1e7, 5e7, 1e20], labels=["SIN_DATO", "<1.7M", "1.7M-10M", "10M-50M", ">50M"]).astype(str),
        "fecha_constitucion": fechas_mixtas(rng, fconst),
        "fecha_inicio_actividades": fechas_mixtas(rng, fconst + rng.integers(0, 200, N).astype("timedelta64[D]"), 0.05, 0.03),
        "departamento": dep_txt,
        "provincia": [f"PROV_{d[:3]}_{rng.integers(1, 6)}" if isinstance(d, str) else np.nan for d in depn],
        "distrito": [f"DIST_{rng.integers(1, 900):03d}" for _ in range(N)],
        "ubigeo": rng.integers(10101, 250401, N).astype(str),
        "zona_comercial": zona_txt,
        "pais": "PERU",
        "estado_contribuyente": np.where(activo, "ACTIVO", rng.choice(["BAJA DE OFICIO", "SUSPENSION TEMPORAL", "BAJA DEFINITIVA"], N)),
        "condicion_domicilio": np.where(rng.random(N) < 0.06, "NO HABIDO", "HABIDO"),
        "flag_cliente_bcp": flag_cli,
        "fecha_alta_cliente": alta,
        "antiguedad_cliente_meses": np.where(cliente, rng.integers(1, 180, N), np.nan),
        "banca": banca,
        "ejecutivo_id": np.where(cliente & (rng.random(N) < 0.8), "EJ" + rng.integers(1, max(2, int(900 * escala)), N).astype(str), ""),
        "flag_exportador": exportador.astype(int),
        "flag_importador": (rng.random(N) < 0.08).astype(int),
        "num_establecimientos": 1 + rng.poisson(pd.Series(segn).map({"MIPE": .1, "PYME": .5, "MEDIANA": 2, "GRANDE": 8, "CORPORATIVO": 30, "ESTADO": 10}).to_numpy()),
        "flag_grupo_economico": (grupo != "").astype(int),
        "id_grupo_economico": grupo,
        "rating_externo": rating,
        "score_buro": score_buro,
        "flag_buen_contribuyente": (rng.random(N) < np.where(np.isin(segn, ["GRANDE", "CORPORATIVO"]), .6, .1)).astype(int),
        "flag_agente_retencion": (rng.random(N) < np.where(es_pagador, .7, .03)).astype(int),
        "flag_email": (rng.random(N) < .7).astype(int),
        "flag_telefono": (rng.random(N) < .85).astype(int),
        "flag_web": (rng.random(N) < np.where(np.isin(segn, ["MIPE"]), .1, .5)).astype(int),
        "fecha_actualizacion": f_act,
        "origen_registro": rng.choice(["SUNAT", "CRM", "CAMPANA_2023", "REFERIDO"], N, p=[.5, .3, .15, .05]),
        "flag_lead_factoring": np.where(lead, rng.choice(["1", "S"], N, p=[.9, .1]), rng.choice(["0", ""], N, p=[.9, .1])),
        "fecha_habilitacion_lead": np.where(lead, (np.datetime64("2025-01-01") + rng.integers(0, 181, N).astype("timedelta64[D]")).astype(str), ""),
        "monto_oferta_factoring": np.where(lead, (mean_inv * rng.uniform(2, 6, N)).round(-2), np.nan),
    })
    # duplicados: misma empresa con registro mas antiguo (hay que quedarse con el mas reciente)
    nd = int(0.015 * N)
    dup = maestro.sample(nd, random_state=seed).copy()
    dup["fecha_actualizacion"] = "2023-01-15"
    dup["segmento"] = ensuciar_texto(rng, dup["segmento"].to_numpy(), SEG_VARIANTES, 1.0)
    dup["flag_cliente_bcp"] = "N"
    maestro = pd.concat([maestro, dup]).sample(frac=1, random_state=seed + 1)
    maestro.to_csv(os.path.join(salida, "maestro_empresas.csv"), index=False)
    print(f"      filas={len(maestro):,} cols={maestro.shape[1]}")
    del maestro, dup

    # ======================================================================
    # TABLA 2: transacciones (por bloques)
    # ======================================================================
    print("[4/6] transacciones")
    path_tx = os.path.join(salida, "transacciones.csv")
    if os.path.exists(path_tx):
        os.remove(path_tx)
    dias_obs = int((FIN_OBS - INI).astype(int)) + 1
    fechas_cal = INI + np.arange(dias_obs).astype("timedelta64[D]")
    mes_cal = pd.DatetimeIndex(fechas_cal).month.to_numpy()
    # pesos estacionales por sector
    pesos_sec = {}
    for s in SEC_NOMBRES + ["ADMINISTRACION_PUBLICA"]:
        pico = SECTORES[s][4] if s in SECTORES else []
        w = np.where(np.isin(mes_cal, pico), 1.8, 1.0); w = np.where(mes_cal == 12, w * 1.15, w)
        pesos_sec[s] = w / w.sum()

    n_tx_edge = rng.poisson(freq * 18)
    n_fact = int(n_tx_edge.sum())
    n_ruido = max(TX_OBJ - n_fact, int(0.2 * TX_OBJ))
    print(f"      pagos de facturas={n_fact:,} | transferencias/ruido={n_ruido:,}")
    glosas_f = np.array(["PAGO FACT {d}", "pago factura {d}", "CANC. {d}", "PAGO PROV {d}", "{d}", "ABONO FACTURA {d}"], dtype=object)
    glosas_r = np.array(["TRANSF", "pago serv", "PRESTAMO SOCIO", "DEVOLUCION", "TRANSFERENCIA ENTRE CTAS", "", "VARIOS"], dtype=object)
    contador = [0]

    def escribir(df):
        df.to_csv(path_tx, mode="a", header=not os.path.exists(path_tx), index=False)

    def armar(o, b, monto, fecha_idx, tipo, plazo, es_fact):
        n = len(o)
        fecha = fechas_cal[fecha_idx]
        hora = rng.integers(7 * 3600, 22 * 3600, n).astype("timedelta64[s]")
        fh = pd.to_datetime(fecha) + pd.to_timedelta(hora)
        fh_txt = fh.strftime("%Y-%m-%d %H:%M:%S").to_numpy(dtype=object)
        ralt = rng.random(n) < 0.04
        fh_txt[ralt] = fh[ralt].strftime("%d/%m/%Y %H:%M").to_numpy(dtype=object)
        usd = rng.random(n) < 0.10
        monto = np.where(usd, monto / TC, monto)
        mon = np.where(usd, rng.choice(["USD", "usd", "US$"], n, p=[.8, .1, .1]), rng.choice(["PEN", "S/", "SOL", "PEN "], n, p=[.85, .08, .04, .03]))
        monto_txt = np.round(monto, 2).astype(object)
        rc = rng.random(n)
        mm = rc < 0.02
        monto_txt[mm] = [f"{v:,.2f}" for v in np.round(monto[mm], 2)]
        monto_txt[(rc >= 0.02) & (rc < 0.0215)] = 999999999.99
        bo = np.where(cliente[o], "002", rng.choice(list(BANCOS_COD.values())[1:], n))
        bb = np.where(cliente[b], "002", rng.choice(list(BANCOS_COD.values())[1:], n))
        bo = np.where((bo == "002") & (rng.random(n) < 0.1), "BCP", bo)
        doc = np.where(es_fact, np.char.add("F00" + rng.integers(1, 9, n).astype(str) + "-", rng.integers(1, 99999, n).astype(str)), "")
        dc = np.where(es_fact, np.round(plazo + rng.normal(0, 6, n)), np.nan)
        dc = np.where(rng.random(n) < 0.25, np.nan, dc)
        femi = np.where(np.isnan(dc), "", (fecha - np.nan_to_num(dc).astype(int).astype("timedelta64[D]")).astype(str))
        glo = np.where(es_fact, [g.format(d=d) for g, d in zip(glosas_f[rng.integers(0, len(glosas_f), n)], doc)], glosas_r[rng.integers(0, len(glosas_r), n)])
        oid = ids[o].astype(str).astype(object)
        bid = ids[b].astype(str).astype(object)
        # inconsistencias de llave: algunos con prefijo EMP
        rp = rng.random(n) < 0.05
        oid[rp] = emp_id(ids[o][rp])
        estado = rng.choice(["PROCESADA", "RECHAZADA", "PENDIENTE"], n, p=[.96, .03, .01])
        lote = np.where(tipo == "PAGO_PROVEEDORES_MASIVO", "L" + rng.integers(1, 10**6, n).astype(str), "")
        idtx = np.arange(contador[0], contador[0] + n) + 10_000_000
        contador[0] += n
        df = pd.DataFrame({
            "id_transaccion": idtx, "fecha_hora": fh_txt, "fecha_valor": fecha.astype(str),
            "id_ordenante": oid, "id_beneficiario": bid, "tipo_transaccion": tipo,
            "canal": rng.choice(["BANCA_ELECTRONICA", "TELECREDITO", "VENTANILLA", "API", "APP"], n, p=[.35, .3, .1, .1, .15]),
            "monto": monto_txt, "moneda": mon, "banco_ordenante": bo, "banco_beneficiario": bb,
            "cuenta_ordenante_hash": hex_ids(rng, n, 10), "cuenta_beneficiario_hash": hex_ids(rng, n, 10),
            "num_documento": doc, "dias_credito_documento": dc, "fecha_emision_documento": femi,
            "estado_tx": estado, "glosa": glo, "flag_masivo": (tipo == "PAGO_PROVEEDORES_MASIVO").astype(int),
            "lote_id": lote, "comision_tx": np.round(rng.choice([0, 3.5, 7, 12], n, p=[.5, .2, .2, .1]), 2),
            "codigo_rpta": np.where(estado == "RECHAZADA", rng.choice(["E01", "E07", "E99"], n), "000"),
        })
        # duplicados exactos (reenvio de archivo)
        d = df.sample(frac=0.008, random_state=int(rng.integers(1e9)))
        return pd.concat([df, d])

    # pagos de facturas por bloques de aristas
    B = 150_000
    for s0 in range(0, E, B):
        sl = slice(s0, min(E, s0 + B))
        nt = n_tx_edge[sl]
        e_rep = np.repeat(np.arange(s0, min(E, s0 + B)), nt)
        if len(e_rep) == 0:
            continue
        sec_e = secn[src[e_rep]]
        fidx = np.empty(len(e_rep), dtype=int)
        for s in np.unique(sec_e):
            mm = sec_e == s
            fidx[mm] = rng.choice(dias_obs, size=mm.sum(), p=pesos_sec[s])
        monto = amt[e_rep] * rng.lognormal(0, 0.45, len(e_rep))
        payer = dst[e_rep]
        tipo = np.where(cliente[payer] & (rng.random(len(e_rep)) < 0.6), "PAGO_PROVEEDORES_MASIVO", "PAGO_FACTURA").astype(object)
        ext = rng.random(len(e_rep)) < 0.005
        tipo[ext] = "EXTORNO"; monto[ext] = -monto[ext]
        escribir(armar(payer, src[e_rep], monto, fidx, tipo, eplazo[e_rep], ~ext))
    # ruido: transferencias entre empresas cualesquiera
    for s0 in range(0, n_ruido, 500_000):
        n = min(500_000, n_ruido - s0)
        o = rng.integers(0, N, n); b = rng.integers(0, N, n)
        tipo = rng.choice(["TRANSFERENCIA_INTERBANCARIA", "TRANSFERENCIA_INTERNA"], n, p=[.6, .4]).astype(object)
        monto = rng.lognormal(np.log(3000), 1.3, n)
        escribir(armar(o, b, monto, rng.integers(0, dias_obs, n), tipo, np.zeros(n), np.zeros(n, bool)))
    print(f"      filas aprox={contador[0]*1.008:,.0f} cols=22")

    # ======================================================================
    # TABLA 3: factoring_operaciones
    # ======================================================================
    print("[5/6] factoring_operaciones")
    # payer principal de cada proveedor (para asignar pagador a la operacion)
    first_payer = np.full(N, -1)
    first_payer[src[::-1]] = dst[::-1]
    rows = []

    def ops(empresas, n_ops, f_ini, f_fin, banco_bcp, estado_forzado=None):
        emp = np.repeat(empresas, n_ops)
        n = len(emp)
        if n == 0:
            return None
        span = int((f_fin - f_ini).astype(int))
        fdes = f_ini + rng.integers(0, span + 1, n).astype("timedelta64[D]")
        pag = first_payer[emp].copy()
        sinpag = pag < 0
        pag[sinpag] = pag_idx[rng.integers(0, len(pag_idx), sinpag.sum())]
        plazo = np.clip(np.where(plazo_de[pag] > 0, plazo_de[pag], 60) + rng.normal(0, 10, n), 15, 180).round()
        nf = 1 + rng.poisson(1.5, n)
        monto = mean_inv[emp] * nf * rng.lognormal(0, 0.3, n)
        tasa = np.clip(rng.normal(1.3, 0.3, n) + 0.002 * (plazo - 60), 0.6, 3.0)
        des = monto * (1 - tasa / 100 * plazo / 30)
        if banco_bcp is True:
            ent = np.array(["BCP"] * n, dtype=object); fuente = np.array(["BCP_INTERNO"] * n, dtype=object)
        else:
            ent = rng.choice(["011", "003", "009", "038", "035"], n, p=[.35, .3, .2, .1, .05]).astype(object)
            fuente = np.array(["SPI"] * n, dtype=object)
        if estado_forzado is not None:
            est = np.array([estado_forzado] * n, dtype=object)
        else:
            vencida = (fdes + plazo.astype(int).astype("timedelta64[D]")) < np.datetime64("2026-01-31")
            est = np.where(vencida, rng.choice(["PAGADO", "VENCIDO"], n, p=[.95, .05]), "DESEMBOLSADO").astype(object)
            est[rng.random(n) < 0.03] = "ANULADO"
        usd = rng.random(n) < 0.12
        tasa_txt = np.round(tasa, 2).astype(object)
        tm = rng.random(n) < 0.15
        tasa_txt[tm] = [f"{v:.2f}%" for v in tasa[tm]]
        return pd.DataFrame({
            "id_proveedor": emp_id(ids[emp]), "ruc_proveedor_hash": "", "id_pagador": emp_id(ids[pag]),
            "fuente_registro": fuente, "entidad_financiera": ent,
            "producto": rng.choice(["FACTORING_ELECTRONICO", "DESCUENTO_FACTURAS", "CONFIRMING"], n, p=[.65, .25, .1]),
            "modalidad": rng.choice(["CON_RECURSO", "SIN_RECURSO"], n, p=[.7, .3]),
            "fecha_registro": (fdes - rng.integers(0, 6, n).astype("timedelta64[D]")).astype(str),
            "fecha_desembolso": fdes.astype(str),
            "fecha_vencimiento": (fdes + plazo.astype(int).astype("timedelta64[D]")).astype(str),
            "plazo_dias": plazo, "num_facturas": nf,
            "monto_facturas": np.round(np.where(usd, monto / TC, monto), 2), "moneda": np.where(usd, "USD", "PEN"),
            "monto_desembolsado": np.round(np.where(usd, des / TC, des), 2), "tasa_mensual": tasa_txt,
            "comision": np.round(rng.choice([0, 25, 50, 80], n), 2), "estado_operacion": est,
            "dias_atraso": np.where(est == "VENCIDO", rng.integers(1, 90, n), 0),
            "canal_origen": rng.choice(["EJECUTIVO", "WEB", "APP", "TELEBANCA", ""], n, p=[.5, .25, .1, .1, .05]),
            "ejecutivo_id": np.where(fuente == "BCP_INTERNO", "EJ" + rng.integers(1, max(2, int(900 * escala)), n).astype(str), ""),
            "flag_primera_op": 0, "sector_pagador_desc": secn[pag],
            "segmento_pagador": segn[pag], "calificacion_pagador": rng.choice(["AAA", "AA", "A", "BBB", "NR"], n),
            "observaciones": rng.choice(["", "", "", "OK", "FACTURA CONFORME", "REPROGRAMADO", "sin obs"], n),
        })

    hb = np.where(hist_bcp)[0]; ho = np.where(hist_otro)[0]
    rows.append(ops(hb, 1 + rng.poisson(3, len(hb)), INI, FIN_OBS, True))
    rows.append(ops(ho, 1 + rng.poisson(3, len(ho)), INI, FIN_OBS, False))
    tg = np.where(target)[0]
    rows.append(ops(tg, n_ops_t[tg], INI_TGT, FIN_TGT, True))
    # ruido en ventana target: otros bancos (no cuentan) y BCP ANULADAS (no cuentan)
    oo = np.where(hist_otro & (rng.random(N) < 0.5))[0]
    rows.append(ops(oo, 1 + rng.poisson(1, len(oo)), INI_TGT, FIN_TGT, False))
    an = np.where(es_prov & ~target & (rng.random(N) < 0.006))[0]
    rows.append(ops(an, np.ones(len(an), int), INI_TGT, FIN_TGT, True, "ANULADO"))
    fac = pd.concat([r for r in rows if r is not None], ignore_index=True)
    # monto target real (en PEN) se recalcula desde las operaciones para consistencia
    # duplicados BCP vistos tambien por SPI con codigo 002
    bcp_rows = fac[fac.fuente_registro == "BCP_INTERNO"].sample(frac=0.15, random_state=seed)
    bcp_rows = bcp_rows.assign(fuente_registro="SPI", entidad_financiera="002", ejecutivo_id="", canal_origen="")
    fac = pd.concat([fac, bcp_rows], ignore_index=True)
    # variaciones de llave
    lw = rng.random(len(fac)) < 0.04
    fac.loc[lw, "id_proveedor"] = fac.loc[lw, "id_proveedor"].str.lower() + " "
    fac.loc[rng.random(len(fac)) < 0.05, "id_pagador"] = np.nan
    fac = fac.sample(frac=1, random_state=seed + 2).reset_index(drop=True)
    fac.insert(0, "id_operacion", "OP" + pd.Series(np.arange(len(fac)) + 1).astype(str).str.zfill(9))
    # primera operacion
    f_ord = fac.sort_values("fecha_desembolso")
    fac.loc[f_ord.groupby(f_ord.id_proveedor.str.strip().str.upper()).head(1).index, "flag_primera_op"] = 1

    # clave: target y monto (PEN) desde operaciones BCP internas validas en ventana
    fd = pd.to_datetime(fac.fecha_desembolso)
    val = (fac.fuente_registro == "BCP_INTERNO") & (fac.estado_operacion != "ANULADO") & (fd >= "2025-07-01") & (fd <= "2025-12-31")
    mpen = np.where(fac.moneda == "USD", fac.monto_facturas * TC, fac.monto_facturas)
    agg = pd.Series(mpen[val.to_numpy()]).groupby(fac.loc[val, "id_proveedor"].str.strip().str.upper().to_numpy()).sum()

    # quitar ventana target para empresas test
    test_ids = set(emp_id(ids[test]))
    prov_norm = fac.id_proveedor.str.strip().str.upper()
    quitar = prov_norm.isin(test_ids) & (fd >= "2025-07-01")
    fac_pub = fac[~quitar]
    fac_pub.to_csv(os.path.join(salida, "factoring_operaciones.csv"), index=False)
    print(f"      filas={len(fac_pub):,} cols={fac_pub.shape[1]} (ocultas para test: {quitar.sum():,})")

    # ======================================================================
    # TABLA 4: perfil_empresa (formato ancho 12 meses; m01 = 2025-06 ... m12 = 2024-07)
    # ======================================================================
    print("[6/6] perfil_empresa")
    t = np.arange(12)  # 0 = m01 (mas reciente)
    base_saldo = pd.Series(segn).map({"MIPE": 8e3, "PYME": 6e4, "MEDIANA": 4e5, "GRANDE": 3e6, "CORPORATIVO": 2e7, "ESTADO": 1e7}).to_numpy() * rng.lognormal(0, 0.8, N)
    tiene_linea = rng.random(N) < np.where(cliente, 0.55, 0.0)
    cols = {}
    for j in range(12):
        mj = f"m{j+1:02d}"
        tendencia = np.exp(0.035 * stress * t[j])            # estresadas: saldo cae hacia el presente
        sal = base_saldo * tendencia * rng.lognormal(0, 0.25, N)
        cols[f"saldo_cta_cte_{mj}"] = np.where(cliente, sal.round(2), np.nan)
        cols[f"saldo_ahorro_{mj}"] = np.where(cliente & (rng.random(N) < 0.6), (0.3 * sal * rng.lognormal(0, 0.5, N)).round(2), np.nan)
        cols[f"uso_linea_pct_{mj}"] = np.where(tiene_linea, np.clip(100 * sigmoid(0.9 * stress - 0.08 * t[j] * stress + rng.normal(0, 0.5, N)), 0, 100).round(1), np.nan)
        deuda = base_saldo * 2.5 * np.exp(0.03 * stress * (11 - t[j])) * rng.lognormal(0, 0.2, N)
        cols[f"deuda_sistema_{mj}"] = np.where(segn == "ESTADO", np.nan, deuda.round(0))
        cols[f"num_tx_abono_{mj}"] = np.where(cliente, rng.poisson(3 + np.bincount(src, minlength=N) * 2), np.nan)
        abonos = vol / 18 * rng.lognormal(0, 0.3, N) + base_saldo * 0.5
        cols[f"monto_abonos_{mj}"] = np.where(cliente, abonos.round(2), np.nan)
        cols[f"monto_cargos_{mj}"] = np.where(cliente, (abonos * (1 + 0.04 * stress) * rng.lognormal(0, 0.1, N)).round(2), np.nan)
        cols[f"dias_atraso_max_{mj}"] = rng.poisson(np.exp(0.6 * stress - 1.5))
    perfil = pd.DataFrame(cols)
    # faltantes y centinelas
    for c in perfil.columns:
        mm = rng.random(N)
        perfil.loc[mm < 0.04, c] = np.nan
        perfil.loc[(mm >= 0.04) & (mm < 0.05), c] = -999
    contacto = np.where(cliente & (rng.random(N) < 0.4), (np.datetime64("2025-01-01") + rng.integers(0, 180, N).astype("timedelta64[D]")).astype(str), "")
    # FUGA: a las empresas target se les registro contacto comercial dentro de la ventana target
    fuga_c = target & (rng.random(N) < 0.55)
    contacto = np.where(fuga_c, (INI_TGT + rng.integers(0, 120, N).astype("timedelta64[D]")).astype(str), contacto)
    flag_fact_activo = ((target & (rng.random(N) < 0.8)) | (hist_bcp & (rng.random(N) < 0.3))).astype(int)  # FUGA
    head = pd.DataFrame({
        "id_empresa": emp_id(ids),
        "periodo_corte": np.where(rng.random(N) < 0.97, "202506", "2025-06"),
        "num_productos_activos": np.where(cliente, 1 + rng.poisson(1.5, N), 0),
        "flag_cta_cte": cliente.astype(int),
        "flag_ahorro": (cliente & (rng.random(N) < .6)).astype(int),
        "flag_prestamo_capital_trabajo": (tiene_linea & (rng.random(N) < .7)).astype(int),
        "flag_leasing": (cliente & (rng.random(N) < np.where(np.isin(secn, ["TRANSPORTE_LOGISTICA", "CONSTRUCCION", "MINERIA"]), .3, .05))).astype(int),
        "flag_carta_fianza": (cliente & (rng.random(N) < np.where(has_estado, .55, .08))).astype(int),
        "flag_tarjeta_empresarial": (cliente & (rng.random(N) < .3)).astype(int),
        "flag_telecredito": (cliente & es_pagador & (rng.random(N) < .8)).astype(int),
        "flag_banca_movil": (cliente & (rng.random(N) < .7)).astype(int),
        "flag_seguros": (cliente & (rng.random(N) < .2)).astype(int),
        "calificacion_sbs": np.select([stress > 2.2, stress > 1.6, stress > 1.1], ["DEF", "CPP", "CPP"], "NOR"),
        "score_riesgo_interno": np.where(cliente, np.clip(500 - 60 * stress + rng.normal(0, 50, N), 0, 1000).round(), np.nan),
        "deuda_total_sistema": cols["deuda_sistema_m01"],
        "num_entidades_sistema": 1 + rng.poisson(1 + 0.5 * np.clip(stress, 0, None), N),
        "linea_credito_aprobada": np.where(tiene_linea, (base_saldo * 3).round(-2), np.nan),
        "fecha_ultimo_contacto_comercial": contacto,
        "resultado_ultimo_contacto": np.where(contacto == "", "", rng.choice(["INTERESADO", "NO_CONTESTA", "NO_INTERESADO", "VOLVER_A_LLAMAR"], N)),
        "flag_factoring_activo": flag_fact_activo,
    })
    perfil = pd.concat([head, perfil], axis=1)
    # solo ~85% de empresas tienen perfil (no todas aparecen)
    perfil = perfil[rng.random(N) < 0.85]
    perfil.to_csv(os.path.join(salida, "perfil_empresa.csv"), index=False)
    print(f"      filas={len(perfil):,} cols={perfil.shape[1]}")

    # ---------------- set de evaluacion y clave ----------------
    tid = emp_id(ids[test])
    pd.DataFrame({"id_empresa": tid}).to_csv(os.path.join(salida, "ids_evaluacion.csv"), index=False)
    grupo_ab = np.where(lead, "A_LEAD", np.where(pyme_cli, "B_PYME_SIN_LEAD", "C_RESTO"))
    clave = pd.DataFrame({"id_empresa": tid, "zona": zona[test], "segmento": segn[test], "sector": secn[test],
                          "lead": lead[test].astype(int), "grupo": grupo_ab[test]})
    clave["target"] = clave.id_empresa.map(agg).notna().astype(int)
    clave["monto_target"] = clave.id_empresa.map(agg).fillna(0.0).round(2)
    clave.to_csv(os.path.join(salida, "_clave_oculta", "clave.csv"), index=False)
    print(f"\nListo. Test: {len(tid):,} empresas | tasa target test: {clave.target.mean():.2%}")
    print(clave.groupby("grupo").agg(n=("target", "size"), tasa=("target", "mean"), monto_medio=("monto_target", "mean")).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--escala", type=float, default=1.0)
    ap.add_argument("--salida", default="data_full")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    generar(a.escala, a.salida, a.seed)
