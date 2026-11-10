"""Compatibilidad con el comando antiguo; usar preferentemente src/paso01_medir.py."""
from pathlib import Path
import runpy
import sys


if __name__ == "__main__":
    src = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(src))
    runpy.run_path(str(src / "paso01_medir.py"), run_name="__main__")
