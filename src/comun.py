"""Utilidades compartidas por todos los pasos.
Todo script importa de aqui las rutas y la conexion: si cambia una ruta, se cambia en un solo lugar."""
import os
import time
from contextlib import contextmanager
import duckdb
import psutil

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(RAIZ, "data")          # CSV originales (nunca se modifican)
LIMPIO = os.path.join(RAIZ, "limpio")      # tablas limpias en parquet (salida del paso 4)
FEAT = os.path.join(RAIZ, "features")      # tablas de variables (salida del paso 7)
SAL = os.path.join(RAIZ, "salidas")        # modelos, predicciones, entrega
for d in (LIMPIO, FEAT, SAL):
    os.makedirs(d, exist_ok=True)

TC = 3.75                                   # tipo de cambio dado en el enunciado
CORTE = "2025-06-30"                        # ultimo dia que "conocemos"
INI_TGT, FIN_TGT = "2025-07-01", "2025-12-31"
CUPOS = {"LIMA": 0.50, "NORTE": 0.18, "SUR": 0.17, "CENTRO": 0.10, "ORIENTE": 0.05}


def csv(nombre):
    return os.path.join(DATA, nombre)


def pq(carpeta, nombre):
    return os.path.join(carpeta, nombre + ".parquet")


def conectar(memoria="4GB"):
    """Conexion DuckDB con limite de memoria y la macro de llave comun.
    memory_limit: si la consulta necesita mas, DuckDB escribe a disco en vez de morir."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memoria}'")
    con.execute("SET threads=4")
    # llave unica para las 4 tablas: 'EMP00001007', '1007', ' emp00001007 ' -> 1007
    con.execute("CREATE OR REPLACE MACRO k(x) AS TRY_CAST(regexp_replace(upper(trim(x)), '^EMP0*', '') AS BIGINT)")
    return con


@contextmanager
def cronometro(nombre):
    t = time.time()
    yield
    rss = psutil.Process().memory_info().rss / 1e9
    print(f"[{nombre}] {time.time() - t:,.1f} s | memoria actual del proceso: {rss:.2f} GB")
