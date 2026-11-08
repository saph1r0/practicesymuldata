"""PASO 3 - Grano y llaves.
Pregunta 1: que representa UNA fila de cada tabla (el 'grano')?
Pregunta 2: la llave es unica donde deberia serlo?
Pregunta 3: las llaves de distintas tablas se pueden cruzar? que % cruza?"""
from comun import csv, conectar, cronometro

con = conectar()
for t in ["maestro_empresas", "transacciones", "factoring_operaciones", "perfil_empresa", "ids_evaluacion"]:
    con.execute(f"CREATE VIEW {t} AS SELECT * FROM read_csv('{csv(t + '.csv')}', all_varchar=true)")

print("1) Unicidad de la llave declarada")
for t, llave in [("maestro_empresas", "id_empresa"), ("transacciones", "id_transaccion"),
                 ("factoring_operaciones", "id_operacion"), ("perfil_empresa", "id_empresa")]:
    n, d = con.execute(f"SELECT count(*), count(DISTINCT {llave}) FROM {t}").fetchone()
    print(f"   {t:<22} filas={n:>10,}  {llave} distintos={d:>10,}  repetidos={n - d:>8,}")

print("\n2) Las filas repetidas del maestro: son identicas o difieren?")
print(con.execute("""
  WITH d AS (SELECT * FROM maestro_empresas WHERE id_empresa IN
             (SELECT id_empresa FROM maestro_empresas GROUP BY 1 HAVING count(*) > 1))
  SELECT id_empresa, fecha_actualizacion, segmento, flag_cliente_bcp FROM d ORDER BY id_empresa, fecha_actualizacion LIMIT 6
""").df().to_string(index=False))
print("   transacciones: filas repetidas que son copia EXACTA en todas las columnas:",
      f"{con.execute('SELECT count(*) - (SELECT count(*) FROM (SELECT DISTINCT * FROM transacciones)) FROM transacciones').fetchone()[0]:,}")

print("\n3) Formatos de las llaves de empresa (patron: L=letra, 9=digito)")
for t, c in [("maestro_empresas", "id_empresa"), ("transacciones", "id_ordenante"), ("transacciones", "id_beneficiario"),
             ("factoring_operaciones", "id_proveedor"), ("perfil_empresa", "id_empresa")]:
    r = con.execute(f"""SELECT regexp_replace(regexp_replace({c}, '[A-Za-z]', 'L', 'g'), '[0-9]', '9', 'g') AS patron,
                               count(*) n FROM {t} GROUP BY 1 ORDER BY 2 DESC LIMIT 4""").df()
    print(f"   {t}.{c}: " + " | ".join(f"'{p}'={n:,}" for p, n in zip(r.patron, r.n)))

print("\n4) Cobertura: que % de cada tabla encuentra su empresa en el maestro")
with cronometro("cruces"):
    con.execute("CREATE TABLE llaves_m AS SELECT DISTINCT k(id_empresa) AS k FROM maestro_empresas")
    for t, c in [("transacciones", "id_ordenante"), ("transacciones", "id_beneficiario"),
                 ("factoring_operaciones", "id_proveedor"), ("perfil_empresa", "id_empresa"), ("ids_evaluacion", "id_empresa")]:
        crudo, limpio = con.execute(f"""
          SELECT avg(({c} IN (SELECT id_empresa FROM maestro_empresas))::INT),
                 avg((k({c}) IN (SELECT k FROM llaves_m))::INT) FROM {t}""").fetchone()
        print(f"   {t}.{c:<16} sin limpiar: {crudo:6.1%}   con llave k(): {limpio:6.1%}")
    n_m = con.execute("SELECT count(*) FROM llaves_m").fetchone()[0]
    n_p = con.execute("SELECT count(DISTINCT k(id_empresa)) FROM perfil_empresa").fetchone()[0]
    print(f"   empresas del maestro con perfil: {n_p / n_m:.1%}  -> usar LEFT JOIN desde el maestro")
