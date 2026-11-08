# Simulacro DataFest BCP – Caso Factoring por Rubros

La guía completa (enunciado, diccionario, plan de 10 h, trampas y pitch) está en el doc "Simulacro DataFest BCP – Caso Factoring por Rubros".

## Inicio rápido

```bash
pip install duckdb lightgbm pandas scikit-learn pyarrow
# Data del concurso: ~8.9M filas, 210 columnas, ~1.8 GB. Unos 3 min y hasta 4.7 GB de RAM.
python generar_datafest_sim.py --escala 1.0 --salida data
# Versión de práctica: ~0.9M filas, 20 s
python generar_datafest_sim.py --escala 0.1 --salida data_lite
```

Con la misma semilla (por defecto 2026) se genera exactamente la misma data. Para un segundo intento: `--seed 7`.

## Evaluar una entrega

```bash
python evaluar.py --entrega salidas/entrega.csv --data data
```

Formato: `id_empresa, prob_factoring, monto_esperado, priorizado` (una fila por id de `ids_evaluacion.csv`).

## Reglas

- Nadie abre `data/_clave_oculta/` ni `solucion_referencia.py` hasta terminar las 10 horas.
- Solo VS Code + GitHub Copilot Chat. Repo en GitHub con una rama por integrante y pull requests.
