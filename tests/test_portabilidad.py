"""Pruebas pequeñas, sin generar ni modificar los datos del equipo."""
import contextlib
import io
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import comun


class PortabilidadTest(unittest.TestCase):
    def test_cronometro_reporta_memoria_actual(self):
        salida = io.StringIO()
        with patch("comun.psutil.Process") as proceso:
            proceso.return_value.memory_info.return_value.rss = 500_000_000
            with contextlib.redirect_stdout(salida), comun.cronometro("prueba"):
                pass
        self.assertIn("memoria actual del proceso: 0.50 GB", salida.getvalue())

    def test_paso01_con_datos_temporales(self):
        tablas = ("maestro_empresas", "transacciones", "factoring_operaciones", "perfil_empresa")
        salida = io.StringIO()
        with tempfile.TemporaryDirectory(prefix="datos con espacios ") as carpeta:
            for tabla in tablas:
                (Path(carpeta) / f"{tabla}.csv").write_text("id,valor\n1,10\n2,20\n", encoding="utf-8")
            with patch.object(comun, "DATA", carpeta), contextlib.redirect_stdout(salida):
                runpy.run_path(str(Path(comun.RAIZ) / "src" / "paso01_medir.py"), run_name="__main__")
        self.assertIn("RAM total", salida.getvalue())
        self.assertIn("TOTAL", salida.getvalue())
        for tabla in tablas:
            self.assertIn(tabla, salida.getvalue())


if __name__ == "__main__":
    unittest.main()
