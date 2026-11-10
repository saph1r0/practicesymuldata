# Simulacro DataFest BCP – Caso Factoring por Rubros

Para ejecutar el proyecto en **Windows, macOS o Linux**, sigue el
[README principal](../README.md). Usa `data/` en la raíz del repositorio para
generar los datos y ejecutar los pasos; no necesitas duplicarlos en esta carpeta.

## Datos y evaluación

Desde la raíz del repositorio, con el entorno virtual activado:

```bash
python simulacro_datafest_codigo/generar_datafest_sim.py --escala 0.1 --salida data
python src/run_all.py pyme
python src/evaluar.py --entrega salidas/entrega_pyme.csv --data data
```

Usa `--escala 1.0` para la data completa: aproximadamente 8.9 millones de filas.
Con la misma semilla (por defecto 2026) se genera la misma data. Para otro intento usa `--seed 7`.

Formato de entrega: `id_empresa, prob_factoring, monto_esperado, habilitar, priorizado`
(una fila por id de `ids_evaluacion.csv`).

## Reglas

- Nadie abre `data/_clave_oculta/` ni `solucion_referencia.py` hasta terminar las 10 horas.
- Solo VS Code + GitHub Copilot Chat. Repo en GitHub con una rama por integrante y pull requests.
