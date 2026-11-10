# Guía paso a paso – código

El mismo código funciona en **Windows, macOS y Linux**. No necesitas WSL.
Cada archivo de `src/` corresponde a un paso de la guía de análisis.

## 1. Abre la terminal en la raíz del proyecto

En VS Code, abre la carpeta `practicesymuldata` y selecciona **Terminal → Nueva terminal**.
Todos los comandos de esta guía se ejecutan allí, donde está este README. No hagas `cd src`.

## 2. Crea y activa el entorno virtual

Necesitas Python 3.11 o posterior. Recomendamos Python 3.12 o 3.13.

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activación, permite scripts solo para esta terminal y vuelve a activarlo:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

**macOS y Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Si Ubuntu/Debian indica que falta `venv`, instala `python3-venv` con tu gestor de paquetes.
En macOS, si LightGBM indica que falta `libomp`, instala OpenMP con `brew install libomp`.

Cuando veas `(.venv)`, instala las dependencias (igual en los tres sistemas):

```bash
python -m pip install -r requirements.txt
```

En una terminal nueva solo necesitas activar el entorno otra vez, no recrearlo.
Cada integrante crea su propio entorno: no se comparte ni se sube a Git.

## 3. Genera los datos de práctica

```bash
python simulacro_datafest_codigo/generar_datafest_sim.py --escala 0.1 --salida data
```

Si ya tienes los CSV en `data/`, omite este paso. No los copies a otra carpeta.
Para la versión completa usa `--escala 1.0`; requiere más tiempo, RAM y espacio.

## 4. Ejecuta el flujo

Primero comprueba que se pueden leer los datos y medir la memoria:

```bash
python src/paso01_medir.py
```

Si termina sin errores, genera la entrega:

```bash
python src/run_all.py pyme
```

El flujo ejecuta los pasos 4, 5, 7, 8, 9, 10 y 11, y se detiene si alguno falla.
Puedes cambiar `pyme` por `soles` o `conversiones`.

## 5. Evalúa y abre el resultado

Solo después de que el flujo termine correctamente:

```bash
python src/evaluar.py --entrega salidas/entrega_pyme.csv --data data
```

El resultado está en **`salidas/entrega_pyme.csv`**. Ábrelo desde el explorador de VS Code;
la ruta del CSV no es un comando de terminal.

## Para estudiar: un paso a la vez

En lugar de `run_all.py`, ejecuta estos comandos **uno por uno y en orden**:

```bash
python src/paso01_medir.py        # tamaño y RAM disponible
python src/paso02_perfilar.py     # columnas y calidad de los datos
python src/paso03_llaves.py       # identificadores y cruces
python src/paso04_limpiar.py      # -> limpio/*.parquet
python src/paso05_target.py       # variable objetivo y demostración de fuga
python src/paso06_eda.py          # hipótesis de negocio
python src/paso07_features.py     # variables para el modelo
python src/paso08_tabla.py        # tabla de entrenamiento
python src/paso09_modelo.py       # regla, logística y LightGBM
python src/paso10_monto.py        # monto y valor esperado
python src/paso11_priorizar.py pyme
python src/paso12_explicar.py     # importancia y explicaciones
python src/evaluar.py --entrega salidas/entrega_pyme.csv --data data
```

**Si un paso falla, detente y revisa el error antes de continuar.**

## Notas de compatibilidad

- `psutil` consulta la RAM en los tres sistemas; no usamos `resource` ni `os.sysconf`.
- El cronómetro muestra la **memoria actual del proceso**, no su máximo histórico.
- El antiguo comando `python simulacro_datafest_codigo/src/paso01_medir.py` usa ahora
  el mismo paso 1 y los mismos datos de la raíz.
- Los datos, entornos y resultados generados quedan fuera de Git.

Para comprobar la compatibilidad sin modificar tus datos:

```bash
python -m unittest discover -s tests -v
```

Estas pruebas también se ejecutan automáticamente en GitHub con Windows, macOS y Linux.
