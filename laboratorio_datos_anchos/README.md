# Laboratorio: datos anchos (500 mil filas x 200 columnas)

Acompaña la pestaña "Profundización: datos anchos" de la guía.

```bash
pip install duckdb pandas pyarrow scikit-learn lightgbm
python gen_ancho.py --filas 500000 --salida lab   # ~1 min, CSV de 584 MB + lab/verdad.csv
python exp_memoria.py                              # pandas vs Parquet vs DuckDB (tiempo y memoria pico); crea el Parquet
python paso_a_perfil.py                            # ficha y clase de cada columna -> lab/ficha_columnas.csv
python paso_b_limpiar.py                           # limpieza por clase -> lab/limpio.parquet
python paso_c_seleccion.py                         # embudo F1..F5 -> lab/seleccionadas.csv (~3 min)
python paso_d_comparar.py                          # Gini en validación 2024 vs real 2025
```

`lab/verdad.csv` dice qué es realmente cada columna: úsenlo solo al final, para comprobar qué detectó cada técnica.
Para practicar más rápido: `--filas 100000`.
