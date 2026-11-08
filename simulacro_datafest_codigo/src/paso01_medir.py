from pathlib import Path
import ctypes

import duckdb
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / ("data_lite" if (PROJECT_DIR / "data_lite").exists() else "data")
TABLE_NAMES = (
    "maestro_empresas",
    "transacciones",
    "factoring_operaciones",
    "perfil_empresa",
)


def ram_gb() -> float:
    class MemoryStatus(ctypes.Structure):
        _fields_ = [
            ("length", ctypes.c_ulong),
            ("memory_load", ctypes.c_ulong),
            ("total_phys", ctypes.c_ulonglong),
            ("avail_phys", ctypes.c_ulonglong),
            ("total_page_file", ctypes.c_ulonglong),
            ("avail_page_file", ctypes.c_ulonglong),
            ("total_virtual", ctypes.c_ulonglong),
            ("avail_virtual", ctypes.c_ulonglong),
            ("avail_virtual_extended", ctypes.c_ulonglong),
        ]

    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    kernel32 = ctypes.windll.kernel32
    kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel32.GlobalMemoryStatusEx.restype = ctypes.c_bool
    if not kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise RuntimeError("No se pudo consultar la memoria RAM de Windows.")
    return status.total_phys / 1e9


def main() -> None:
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de datos: {DATA_DIR}")

    ram = ram_gb()
    print(f"RAM total de esta maquina: {ram:.1f} GB")
    print(f"Datos: {DATA_DIR}")
    print()
    print(f"{'tabla':28} {'filas':>12} {'MB en disco':>14} {'GB en pandas (estimado)':>24}")

    con = duckdb.connect()
    total_estimated_gb = 0.0

    for table in TABLE_NAMES:
        csv_path = DATA_DIR / f"{table}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Falta el archivo CSV: {csv_path}")

        escaped_path = str(csv_path).replace("'", "''")
        row_count = con.execute(
            f"SELECT count(*) FROM read_csv('{escaped_path}', all_varchar=true)"
        ).fetchone()[0]
        sample = pd.read_csv(csv_path, nrows=50_000)
        estimated_gb = (
            sample.memory_usage(deep=True).sum() / max(len(sample), 1) * row_count / 1e9
        )
        size_mb = csv_path.stat().st_size / 1_000_000
        total_estimated_gb += estimated_gb

        print(
            f"{table:28} {row_count:12,d} {size_mb:14,.1f} "
            f"{estimated_gb:24,.2f}"
        )

    con.close()
    required_gb = total_estimated_gb * 2.5
    ratio = required_gb / ram

    print(f"{'TOTAL':28} {'':12} {'':14} {total_estimated_gb:24,.2f}")
    print()
    print("Regla practica: pandas necesita 2 a 3 veces este tamano para joins y groupby.")
    print(f"=> se necesitarian aprox. {required_gb:.1f} GB y la maquina tiene {ram:.1f} GB.")

    if total_estimated_gb < ram / 4:
        print("Decision: pandas es suficiente.")
    elif total_estimated_gb < ram / 2:
        print("Decision: comprobar RAM durante joins; usar DuckDB si aparecen picos.")
    else:
        print("Decision: usar DuckDB para tablas grandes y pandas solo para tablas pequenas.")


if __name__ == "__main__":
    main()
