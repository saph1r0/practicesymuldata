# Guía paso a paso – código

Cada archivo de `src/` corresponde a un paso del doc "Guía paso a paso: de 8.9M de filas crudas a la solución".

## Preparación

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install duckdb pandas pyarrow scikit-learn lightgbm
# generar la data del simulacro dentro de data/ (script del zip del simulacro)
python generar_datafest_sim.py --escala 1.0 --salida data
```

## Para estudiar: un paso a la vez

```bash
cd src
python paso01_medir.py       # ¿cabe en memoria?
python paso02_perfilar.py    # qué hay en cada columna
python paso03_llaves.py      # grano, llaves y cruces
python paso04_limpiar.py     # -> limpio/*.parquet
python paso05_target.py      # -> features/target.parquet (+ demostración de fuga)
python paso06_eda.py         # hipótesis de negocio
python paso07_features.py    # -> features/f_*.parquet
python paso08_tabla.py       # -> features/tabla_modelo.parquet
python paso09_modelo.py      # regla vs logística vs LightGBM
python paso10_monto.py       # monto y valor esperado
python paso11_priorizar.py pyme    # o: soles | conversiones  -> salidas/entrega_<criterio>.csv
python paso12_explicar.py    # importancia, SHAP, Gini por rubro
python evaluar.py --entrega ../salidas/entrega_pyme.csv --data ../data
```

## Todo con un comando

```bash
python src/run_all.py pyme     # ~3 min en una laptop de 8 GB con la data completa
```

Funciona igual con la data de práctica (`--escala 0.1`): solo cambia `data/`.
