# Guía paso a paso: de 8.9M de filas crudas a la solución

Nov 7, 2026 · @soppp

## Cómo leer esta guía

La guía resuelve el simulacro completo en 13 pasos, en el orden en que se hace en un equipo de analytics real. Todo el código se ejecutó sobre la data de escala 1.0 (8.9 millones de filas, 213 columnas) en una máquina de 8 GB de RAM, y las salidas que ven aquí son las reales. De punta a punta, el pipeline completo tarda 2.9 minutos.

Cada paso tiene la misma estructura:

- **Qué hacemos**: la acción en una frase.
- **Por qué**: la justificación, que es lo que defienden ante el jurado.
- **Código**: la parte esencial. El archivo completo está en `src/` con el mismo nombre.
- **Qué salió**: la salida real y cómo leerla.
- **Para cualquier otro caso**: qué se mantiene igual si mañana la data es otra.

**Mapa del proceso.**

| Paso | Pregunta que responde | Entra | Sale | Tiempo real |
| --- | --- | --- | --- | --- |
| 0. Entorno | ¿Todos trabajamos igual? | Nada | Repo, carpetas, librerías | 15 min (una vez) |
| 1. Medir | ¿Cabe en memoria? ¿Qué herramienta uso? | CSV | Tamaños y decisión | 30 s |
| 2. Perfilar | ¿Qué hay en cada columna? | CSV | Lista de problemas | 6 s |
| 3. Llaves | ¿Qué es una fila? ¿Las tablas cruzan? | CSV | Regla de llave única | 10 s |
| 4. Limpiar | ¿Cómo dejo cada tabla confiable? | CSV (1.8 GB) | `limpio/*.parquet` (254 MB) | 49 s |
| 5. Target | ¿Qué predigo y con qué fecha de corte? | factoring limpio | `target.parquet` | 2 s |
| 6. EDA | ¿Se cumplen nuestras hipótesis? | limpio + target | Hallazgos para las láminas | 5 s |
| 7. Variables | ¿Cómo resumo millones de eventos por empresa? | limpio | `features/f_*.parquet` | 7 s |
| 8. Tabla de modelado | ¿Está todo unido y sano? | features | `tabla_modelo.parquet` (400 mil × 85) | 3 s |
| 9. Probabilidad | ¿Quién va a convertir? | tabla | Probabilidades + Gini | 65 s |
| 10. Monto | ¿Cuánto va a financiar? | tabla | Montos + R² | 47 s |
| 11. Habilitar y priorizar | ¿A quién llamamos? | probabilidad + monto | `entrega.csv` | 1 s |
| 12. Explicar | ¿Por qué el modelo dice eso? | modelo | Historia para el jurado | 40 s |
| 13. Pipeline | ¿Se puede repetir con un comando? | todo | `run_all.py` | 2.9 min total |

**Una idea para tener en la cabeza todo el tiempo:** la data cruda es grande (8.9 millones de filas), pero la pregunta es por empresa (400 mil). Casi todo el trabajo consiste en pasar de eventos a empresas sin perder información y sin mirar el futuro.

**Cómo estudiarla.** Lean un paso, ejecuten su script, comparen su salida con la de la guía y respondan la pregunta de cierre de ese paso sin mirar. Si su salida es distinta, ahí está el aprendizaje: busquen por qué.

## Paso 0 — Preparar el entorno

**Qué hacemos.** Un ambiente de Python aislado e idéntico en las 5 laptops, un repo con carpetas fijas y un archivo `comun.py` con las rutas y constantes.

**Por qué.** El error más caro de un hackathon no es de modelo: es que un script corre en una laptop y en otra no. Un ambiente aislado (`venv`) evita choques de versiones con lo que ya tengan instalado. Las carpetas fijas hacen que cada integrante sepa dónde dejar y dónde buscar. `comun.py` hace que una ruta o el tipo de cambio se cambien en un solo lugar.

```bash
python -m venv .venv
source .venv/bin/activate            # en Windows: .venv\Scripts\activate
pip install duckdb pandas pyarrow scikit-learn lightgbm
pip freeze > requirements.txt        # congela versiones: los 5 instalan exactamente esto
```

**Por qué cada librería.**

| Librería | Para qué | Por qué esta y no otra |
| --- | --- | --- |
| DuckDB | Leer, limpiar, unir y agregar los CSV grandes con SQL | Procesa por partes sin cargar todo en RAM y escribe a disco si le falta memoria. Spark exige un clúster; SQLite guarda por filas y es mucho más lento para agregaciones grandes |
| pandas | Tablas chicas (menos de 1 millón de filas) y texto | Es lo que todos conocen y tiene el mejor manejo de texto. Polars también sirve, pero aprender otra API el día del concurso es riesgo |
| pyarrow | Leer y escribir Parquet | Es el motor que usan pandas y DuckDB para Parquet |
| scikit-learn | Validación cruzada, métricas, regresión logística | Es el estándar; todo el mundo lee su código |
| LightGBM | Modelo principal | Estándar en banca para datos tabulares: maneja nulos y categorías sin preprocesar y es rápido en CPU. XGBoost da resultados parecidos, pero es más lento |

**Estructura del repo.**

```text
datafest/
  data/        CSV originales. Nunca se modifican. Van en .gitignore
  limpio/      tablas limpias en Parquet (paso 4)
  features/    variables por empresa (pasos 5, 7 y 8)
  salidas/     predicciones, importancias y entrega.csv
  src/         comun.py y pasoNN_*.py: un archivo por paso
  docs/        apuntes y guías (Copilot puede leerlos)
  README.md
```

La idea clave es que cada carpeta es la salida de un paso y la entrada del siguiente. Si el paso 9 falla, no hay que repetir los pasos 1 a 8: sus resultados ya están en disco.

**Para cualquier otro caso.** Esto no cambia nunca: ambiente aislado, datos crudos intocables, una carpeta por etapa y constantes en un solo archivo.

**Pregunta de cierre:** ¿por qué la carpeta `data/` va en `.gitignore`? (Porque pesa 1.8 GB, GitHub rechaza archivos de más de 100 MB y la data no es suya para publicar.)

## Paso 1 — Medir el problema antes de abrirlo

**Qué hacemos.** Antes de abrir un solo archivo medimos tres cosas: la RAM de la máquina, el tamaño de cada archivo y cuánto ocuparía en pandas.

**Por qué.** El reflejo de todo estudiante es `pd.read_csv("transacciones.csv")`. Con 1.5 GB eso puede congelar la laptop por 10 minutos o matar Python sin mensaje claro, y en un hackathon 10 minutos perdidos duelen. Medir toma 30 segundos y decide la herramienta con números.

```python
# src/paso01_medir.py (lo esencial)
ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
filas = con.execute("SELECT count(*) FROM read_csv('data/transacciones.csv', all_varchar=true)").fetchone()[0]
muestra = pd.read_csv("data/transacciones.csv", nrows=50_000)          # solo 50 mil filas
gb = muestra.memory_usage(deep=True).sum() / len(muestra) * filas / 1e9  # extrapolar
```

El truco es leer solo 50 mil filas con pandas, medir cuánto pesa cada fila en memoria (`deep=True` cuenta el texto de verdad) y multiplicar por el total de filas, que DuckDB cuenta sin cargar nada.

**Qué salió (real).**

```text
RAM total de esta maquina: 8.4 GB

tabla                    filas      MB en disco   GB en pandas (estimado)
maestro_empresas         406,000          125            0.24
transacciones          7,946,835        1,516            2.58
factoring_operaciones    219,549           46            0.08
perfil_empresa           340,120          166            0.32
TOTAL                                                    3.23

Regla practica: pandas necesita 2 a 3 veces ese tamano para hacer joins y groupby.
=> se necesitarian ~8 GB y la maquina tiene 8 GB.
```

**Cómo leerlo.** Las 4 tablas caben "justo" en pandas (3.2 GB), pero un join o un groupby crea copias intermedias y la cuenta llega a unos 8 GB: el límite exacto de la máquina, con el sistema operativo, el navegador y VS Code también usando RAM. Conclusión: **transacciones se trabaja con DuckDB; las otras tres, que son chicas, pueden ir con pandas.** Fíjense también en que transacciones son 1.5 GB en disco pero 2.6 GB en pandas: el texto en memoria pesa más que en el archivo.

**Por qué DuckDB resuelve esto.** Trabaja por columnas y por bloques: para sumar `monto` lee solo esa columna, en pedazos, y nunca necesita las 22 columnas de 7.9 millones de filas a la vez. Contar las filas de las 4 tablas (8.9 millones) le tomó 22 segundos usando 0.43 GB.

**Para cualquier otro caso.** La regla de decisión:

| Tamaño estimado en pandas vs RAM | Herramienta |
| --- | --- |
| Menos de 1/4 de la RAM | pandas sin miedo |
| Entre 1/4 y 1/2 | pandas solo para la tabla final; DuckDB para joins y agregaciones |
| Más de 1/2 | DuckDB para todo hasta tener una fila por entidad |
| Más que el disco o cientos de GB | Spark o una base de datos en la nube (no aplica al concurso) |

**Pregunta de cierre:** si el día 13 la tabla grande pesa 4 GB en disco, ¿qué harían antes de escribir cualquier otra línea? (Medir con el script del paso 1 y no abrirla con pandas.)

## Paso 2 — Primer contacto: perfilar cada columna sin cargar la tabla

**Qué hacemos.** Para cada columna de cada tabla obtenemos mínimo, máximo, valores distintos y % de nulos, y listamos todos los valores de las columnas que tienen pocos.

**Por qué.** Antes de limpiar hay que saber qué está sucio. Mirar `head()` muestra 5 filas de 7.9 millones: los problemas casi nunca están ahí. El perfil revisa el 100% de las filas en una pasada.

```python
# src/paso02_perfilar.py (lo esencial)
src = "read_csv('data/transacciones.csv', all_varchar=true)"
con.execute(f"SUMMARIZE SELECT * FROM {src}").df()     # una fila por columna: min, max, distintos, % nulos
con.execute(f"SELECT moneda, count(*) FROM {src} GROUP BY 1 ORDER BY 2 DESC").df()
```

**Por qué `all_varchar=true`.** DuckDB adivina el tipo de cada columna mirando las primeras filas. Si en la fila 3 millones aparece "1,234.50" en una columna que adivinó como número, la carga falla o convierte ese valor a nulo sin avisar. Leyendo todo como texto no se pierde nada, y la conversión la hacemos nosotros en el paso 4, donde vemos qué se pierde.

**Qué salió (extracto real).** SUMMARIZE sobre 7.9 millones de filas tomó 5 segundos.

```text
  column_name            min            max   approx_unique  null_percentage
  id_ordenante           1000005        EMP02800972   486624        0.00   <- dos formatos de llave
  monto                  -1,011.14      999999999.99  5576858       0.00   <- texto con comas, negativos, centinela
  moneda                 PEN            usd           7             0.00   <- 7 formas de escribir 2 monedas
  dias_credito_documento -0.0           99.0          180          38.62   <- 39% nulo
  fecha_hora             01/01/2024 07:02  31/12/2024 21:59  ...    0.00   <- formato dd/mm mezclado

  valores de moneda: 'PEN'=6,080,398, 'USD'=635,201, 'S/'=571,833, 'SOL'=284,633,
                     'PEN '=215,238, 'usd'=79,803, 'US$'=79,729
  valores de banco_ordenante: '002'=4,974,532, 'BCP'=551,740, '009'=404,531, ...
  valores de flag_cliente_bcp (maestro): 'N'=123,465, 'S'=114,989, 'SI'=39,118, nan=31,375,
                     '0'=29,237, 'NO'=29,235, '1'=28,793, 'Y'=9,788
  rubro_descripcion (maestro): 68 valores distintos | pais: 1 valor (constante)
```

**Cómo leerlo: la lista de tareas para el paso 4.**

| Señal en el perfil | Qué significa | Tarea |
| --- | --- | --- |
| min/max con formatos distintos en una llave | Llaves en varios formatos | Normalizar llave (paso 3) |
| `moneda` con 7 valores | Mismo concepto escrito distinto | Mapear a PEN/USD y convertir a soles |
| `monto` con máximo 999999999.99 | Valor centinela ("no sé"), no un pago real | Filtrar |
| `monto` con mínimo negativo | Extornos (reversas) | Excluir del volumen |
| Columna con 1 valor distinto (`pais`) | No aporta información | Descartar |
| 68 valores para un concepto de 16 (`rubro_descripcion`) | Texto libre | Usar otra fuente (CIIU) |
| % de nulos alto (`dias_credito` 39%) | Dato que falta | Decidir: imputar o dejar nulo |

**Cuidado con una trampa del propio perfil.** Como todo se leyó como texto, min y max se comparan como texto: `plazo_dias` muestra mínimo "100.0" y máximo "99.0" porque el texto "1" va antes que "9". No es un error de la data: es un efecto de leer como texto. Para rangos numéricos verdaderos, hay que convertir antes.

**Para cualquier otro caso.** Perfilar todo antes de tocar nada es universal. Las 4 preguntas por columna son siempre las mismas: tipo real, % de nulos, distintos, extremos. Y las columnas con menos de unos 12 valores distintos se miran enteras, porque ahí están las variantes sucias.

**Pregunta de cierre:** una columna `edad` tiene mínimo -1 y máximo 999. ¿Qué son esos valores y qué hacen con ellos? (Centinelas de "no se sabe": se convierten en nulo, nunca se promedian.)

## Paso 3 — Grano y llaves: qué es una fila y cómo se unen las tablas

**Qué hacemos.** Respondemos tres preguntas por tabla: qué representa una fila (el **grano**), si la llave es única donde debería serlo, y qué porcentaje de filas encuentra su pareja en otra tabla.

**Por qué.** Un join con llaves mal entendidas no da error: devuelve filas duplicadas o pierde filas en silencio, y el modelo aprende de una tabla equivocada. Es el bug más común y el más difícil de ver después.

**El grano de cada tabla.**

| Tabla | Una fila es… | Llave | Se une al maestro por |
| --- | --- | --- | --- |
| maestro\_empresas | una empresa | id\_empresa | — |
| transacciones | un pago o transferencia (evento) | id\_transaccion | id\_ordenante (paga) e id\_beneficiario (cobra) |
| factoring\_operaciones | una operación de factoring (evento) | id\_operacion | id\_proveedor (y id\_pagador) |
| perfil\_empresa | una empresa al corte de junio 2025 | id\_empresa | id\_empresa |

```sql
-- src/paso03_llaves.py (lo esencial)
SELECT count(*), count(DISTINCT id_empresa) FROM maestro_empresas;            -- ¿la llave es única?
-- ¿qué formatos tiene la llave? L = letra, 9 = dígito
SELECT regexp_replace(regexp_replace(id_ordenante, '[A-Za-z]', 'L', 'g'), '[0-9]', '9', 'g') AS patron,
       count(*) FROM transacciones GROUP BY 1;
-- una sola llave para todas las tablas
CREATE MACRO k(x) AS TRY_CAST(regexp_replace(upper(trim(x)), '^EMP0*', '') AS BIGINT);
```

**Qué salió (real).**

```text
1) Unicidad de la llave
   maestro_empresas   filas=406,000    id_empresa distintos=400,000      repetidos=6,000
   transacciones      filas=7,946,835  id_transaccion distintos=7,883,765 repetidos=63,070
   factoring          filas=219,549    repetidos=0  |  perfil  filas=340,120  repetidos=0

2) Repetidos del maestro: ¿idénticos o distintos?
   id_empresa   fecha_actualizacion  segmento  flag_cliente_bcp
   EMP00001028  2023-01-15           Mipe      N
   EMP00001028  2025-01-24           MIPE      S          <- el registro nuevo dice que SÍ es cliente
   transacciones: repetidos que son copia EXACTA en todas las columnas: 63,070

3) Formatos de la llave
   transacciones.id_ordenante: '9999999'=4,873,366 | '999999'=2,401,871 | 'LLL99999999'=397,491
   factoring.id_proveedor:     'LLL99999999'=210,777 | 'LLL99999999 '=8,772   <- espacio al final

4) Cobertura contra el maestro      sin limpiar    con llave k()
   transacciones.id_ordenante          5.0%          100.0%
   transacciones.id_beneficiario       0.0%          100.0%
   factoring.id_proveedor             96.0%          100.0%
   empresas del maestro con perfil: 85.0%  -> usar LEFT JOIN desde el maestro
```

**Cómo leerlo y qué decidimos.**

1. **Maestro: los repetidos no son idénticos.** Son versiones de la misma empresa en distintas fechas. Decisión: quedarse con la de `fecha_actualizacion` más reciente, porque es lo último que el banco sabe. Si se quedan con la primera que aparezca, en este ejemplo marcarían como no cliente a un cliente.
2. **Transacciones: los repetidos sí son copias exactas.** Es un archivo reenviado. Decisión: `SELECT DISTINCT *`. Si no, el volumen de esas empresas sale inflado.
3. **Una sola llave entera `k`.** Sin normalizar, solo 0–5% de las transacciones encuentran su empresa: un join directo devolvería una tabla casi vacía sin dar ningún error. Con `k()` cruza el 100%. Se usa un entero y no texto porque compara y une más rápido y ocupa menos memoria.
4. **El perfil cubre el 85%.** Decisión: siempre `LEFT JOIN` partiendo del maestro. Con `INNER JOIN` desaparecen 60 mil empresas, justamente los no clientes, que son los que hay que habilitar.

**Para cualquier otro caso.** Antes de cualquier join: (1) escribe en una frase qué es una fila, (2) verifica `count(*)` contra `count(DISTINCT llave)`, (3) mira los patrones de la llave y (4) mide la cobertura del cruce. Y después de cada join, verifica que el número de filas sea el que esperabas.

**Pregunta de cierre:** después de unir maestro (400 mil) con perfil, obtienen 412 mil filas. ¿Qué pasó? (La llave del perfil tiene repetidos: el join multiplicó filas.)

## Paso 4 — Limpiar cada tabla una sola vez y guardarla en Parquet

**Qué hacemos.** Cada tabla pasa por sus reglas de limpieza y se guarda en `limpio/` en formato Parquet. Después de este paso, nadie vuelve a leer los CSV.

**Por qué limpiar una sola vez.** Si cada integrante limpia por su cuenta, aparecen 5 versiones de "monto en soles" y los números no cuadran entre láminas. Una tabla limpia compartida es la única verdad del equipo.

**Por qué Parquet.** Es columnar y comprimido, y guarda los tipos (fecha, número, texto). Resultado real: 1.8 GB de CSV quedaron en 254 MB de Parquet, se leen en una fracción del tiempo y nadie vuelve a convertir texto a número.

**Regla de herramienta.** Tablas grandes en SQL dentro de DuckDB; tablas chicas (menos de 1 millón de filas) en pandas, que tiene mejor manejo de texto. Cada regla imprime cuántas filas toca: si una regla elimina el 40% de la data, algo está mal y hay que verlo antes de seguir.

### 4.1 Maestro (400 mil filas, pandas)

```python
# deduplicar en SQL: me quedo con el registro más reciente de cada empresa
m = con.execute("""SELECT * EXCLUDE rn FROM (
      SELECT *, k(id_empresa) AS k,
             row_number() OVER (PARTITION BY k(id_empresa) ORDER BY fecha_actualizacion DESC) AS rn
      FROM read_csv('data/maestro_empresas.csv', all_varchar=true)) WHERE rn = 1""").df()

# rubro: el CIIU es más confiable que el texto libre
m["ciiu"] = m.ciiu_rev4.fillna("").str.replace(".0", "", regex=False).str.zfill(4)
mapa = (pd.DataFrame({"c": m.ciiu, "s": m.sector_economico_sbs}).dropna()
        .groupby("c").s.agg(lambda x: x.mode()[0]))      # sector más frecuente de cada CIIU
m["sector"] = m.ciiu.map(mapa).fillna(m.sector_economico_sbs).fillna("DESCONOCIDO")
```

`row_number() OVER (PARTITION BY … ORDER BY …)` es una **función de ventana**: numera las filas dentro de cada empresa, de la más reciente a la más antigua, y nos quedamos con la número 1. Es la forma estándar de deduplicar "quedándose con la última versión".

El rubro se reconstruye así: la columna homologada (`sector_economico_sbs`) está limpia pero tiene 30% de nulos, y el CIIU casi nunca falta. Con las filas que tienen ambos, aprendemos qué sector corresponde a cada CIIU y lo aplicamos a todas. El texto libre de 68 variantes no se usa.

```text
  dedup: 406,000 -> 400,000 filas (se queda el registro mas reciente)
  segmento: 14 variantes -> 6 valores: ['CORPORATIVO','ESTADO','GRANDE','MEDIANA','MIPE','PYME']
  sector: texto libre tenia 68 variantes -> 17 sectores; desconocidos: 0.87%
  zona: desde departamento 98.0%; desconocida final 0.24%
  ventas_anuales_soles: 11,992 centinelas -> nulo
  num_trabajadores: 31,999 centinelas -> nulo
  fecha_constitucion: 4,086 con 1900-01-01 -> nulo; no parseables: 0
  -> limpio/maestro.parquet (400000, 25)          [15 s, 1.7 GB pico]
```

### 4.2 Transacciones (7.9 millones, DuckDB)

```sql
CREATE TABLE tx AS SELECT DISTINCT * FROM tx_raw;                         -- 1) copias exactas fuera
CREATE TABLE txl AS SELECT
  k(id_ordenante) AS ord, k(id_beneficiario) AS ben,
  COALESCE(try_strptime(fecha_hora, '%Y-%m-%d %H:%M:%S'),
           try_strptime(fecha_hora, '%d/%m/%Y %H:%M')) AS fh,              -- 2) dos formatos de fecha
  TRY_CAST(replace(monto, ',', '') AS DOUBLE)                              -- 3) "1,234.50" -> 1234.50
    * CASE WHEN upper(trim(moneda)) IN ('USD','US$') THEN 3.75 ELSE 1 END AS monto_pen,  -- 4) a soles
  ...
COPY (SELECT ... FROM txl WHERE estado_tx = 'PROCESADA' AND fh IS NOT NULL
      AND monto_pen > 0 AND monto_pen <= 5e8)                              -- 5) sin extornos ni centinelas
TO 'limpio/transacciones.parquet' (FORMAT parquet);
```

`try_strptime` y `TRY_CAST` devuelven nulo en vez de romper la consulta cuando un valor no se puede convertir. `COALESCE` prueba el primer formato y, si da nulo, el segundo. Después se cuentan los nulos para confirmar que no se perdió nada.

```text
  crudo 7,946,835 -> sin duplicados 7,883,765 -> PROCESADA 7,568,778 -> final 7,526,397
  fechas no parseadas 0 | montos no numericos 0 | > 500 M 11,849 | <= 0 32,320
  [16.5 s, 4.2 GB pico]
```

Cada flecha es una decisión que se puede explicar: −63 mil copias, −314 mil rechazadas o pendientes (no movieron dinero), −12 mil centinelas y −32 mil extornos. En total se eliminó el 5.3%, una proporción razonable.

El pico de 4.2 GB lo causa `SELECT DISTINCT *`, que compara las 22 columnas. Si su laptop tiene poca RAM, el `memory_limit` de `comun.py` hace que DuckDB use el disco en vez de caerse: va más lento, pero no se pierde el trabajo.

### 4.3 Factoring (220 mil filas, DuckDB)

```sql
SELECT k(id_proveedor) AS k,
  CASE WHEN entidad_financiera IN ('BCP', '002') THEN 'BCP' ELSE 'OTRO' END AS banco,
  ...
FROM factoring WHERE NOT (fuente_registro = 'SPI' AND entidad_financiera = '002')
```

002 es el código de entidad del BCP. Las operaciones BCP aparecen dos veces: una en el registro interno y otra vista por el SPI. Si no se quita la copia, los montos y el target del BCP salen inflados. Resultado: 93,636 operaciones BCP internas y 111,941 de otros bancos.

### 4.4 Perfil (340 mil × 116, pandas)

Tres decisiones: (1) `-999` pasa a nulo (326,869 celdas); (2) los números se guardan en `float32`, que usa la mitad de memoria que `float64` y sobra para saldos; (3) se **eliminan tres columnas con fuga** (`flag_factoring_activo`, `fecha_ultimo_contacto_comercial`, `resultado_ultimo_contacto`) porque contienen información posterior al corte. El paso 5 explica por qué eso destruye un modelo.

**Para cualquier otro caso.** La lista de limpieza es casi siempre la misma: duplicados (exactos o versiones), llaves, tipos, fechas en varios formatos, unidades y monedas, centinelas, categorías escritas de varias formas, filas que no representan el fenómeno (rechazadas, anuladas, reversas) y columnas con fuga. Lo que cambia es el detalle.

**Pregunta de cierre:** ¿por qué las transacciones rechazadas se eliminan en vez de contarlas como facturas? (Porque no movieron dinero: no son evidencia de que alguien le pague a la empresa.)

## Paso 5 — Línea de tiempo, target y separación sin fugas

**Qué hacemos.** Fijamos la fecha de corte (30/06/2025), definimos el target en la ventana posterior y separamos train y test.

**Por qué este es el paso más importante.** Un modelo de negocio siempre se usa en un momento dado para predecir algo que todavía no pasó. Si una variable usa información posterior a ese momento, el modelo "hace trampa": en validación se ve excelente y en la vida real (o en el set ciego) falla. Eso se llama **fuga de información** (*data leakage*) y es la principal razón por la que un equipo con buen Gini en su laptop pierde en la evaluación.

```text
 2024-01-01                       2025-06-30 | 2025-07-01            2025-12-31
 |--------- OBSERVACIÓN: variables ---------| |---- TARGET: ¿hizo factoring BCP? ----|
  transacciones, perfil, factoring histórico      solo factoring (y oculto para test)
```

```sql
-- src/paso05_target.py
CREATE TABLE tgt AS
SELECT k, 1 AS target, sum(monto_pen) AS monto_target
FROM 'limpio/factoring.parquet'
WHERE banco = 'BCP' AND estado <> 'ANULADO'
  AND fecha_desembolso BETWEEN '2025-07-01' AND '2025-12-31'
GROUP BY k;
-- luego: LEFT JOIN desde el maestro, y quien no aparece tiene target = 0
```

**Qué salió (real).**

```text
   transacciones: 2024-01-01 a 2025-06-30
   factoring    : 2024-01-01 a 2025-12-31   <- la única tabla que contiene el futuro

   train 280,156 empresas, tasa target 5.83% | test 119,844 empresas
   en test, empresas con target visible (debe ser 0): 0

   grupo               empresas   tasa    monto medio si convierte (S/)
   A: con lead            9,306   33.1%     305,910
   B: Pyme sin lead     204,294    2.9%     154,102
   C: resto              66,556   11.0%   2,259,235
```

Tres lecturas. La tasa global es 5.8%: el problema está **desbalanceado** (por cada empresa que convierte, 16 no lo hacen), y eso decide la métrica y la validación del paso 9. Los leads convierten 11 veces más que las Pyme sin lead. Y el grupo C (medianas y grandes) convierte poco, pero por montos 7 veces mayores: de ahí sale el debate del criterio de valor del paso 11.

**La demostración de fuga (real).** La misma variable, "número de operaciones BCP", calculada con y sin el filtro de fecha:

```text
   variable    Gini en train   media en train   media en test
   ops_ok          0.295            0.127            0.129     <- con filtro <= 2025-06-30
   ops_fuga        0.980            0.264            0.129     <- sin filtro
```

`ops_fuga` tiene un Gini de 0.98 en train: parece la mejor variable del mundo. Pero miren las medias: en train vale el doble que en test, porque en train incluye las operaciones del target (el futuro) y en test esas operaciones fueron borradas. El modelo aprendería "si tiene muchas operaciones, convierte" y en el set ciego esa señal no existe. Con el filtro, las medias de train y test son iguales (0.127 y 0.129): esa es la prueba de que la variable es honesta.

**Tres pruebas contra la fuga que deben hacer siempre.**

1. **Prueba de fechas:** toda variable se calcula con `fecha <= corte`. Escriban el filtro en cada consulta, aunque crean que la tabla no llega más allá.
2. **Prueba de medias:** compara la media de cada variable en train y en test. Si difieren mucho y no hay una razón de negocio, sospecha.
3. **Prueba del Gini demasiado bueno:** si una sola variable da un Gini de 0.9 o más, o el modelo pasa de 0.93, busca la fuga antes de celebrar.

**Para cualquier otro caso.** Toda predicción tiene un "hoy". En 2024 (efectivo en agencias), el corte era el día en que se planifica el transporte de caudales y el target eran los retiros de los días siguientes. En 2025 (mejor canal), el corte era antes de la campaña y el target la respuesta a ella. Pregunten siempre: ¿en qué fecha se usaría este modelo y qué sabía el banco ese día?

**Pregunta de cierre:** una tabla trae la columna `estado_cuenta_actual`. ¿La pueden usar? (Solo si "actual" significa al 30/06/2025. Si significa "hoy", es fuga: pregunten al mentor.)

## Paso 6 — EDA guiado por hipótesis

**Qué hacemos.** Tomamos cada hipótesis de negocio escrita en la planificación y la contestamos con una tabla de tasas de conversión. Solo con train.

**Por qué así y no "explorar todo".** Con 213 columnas, explorar sin rumbo produce 200 gráficos y ninguna conclusión. Una hipótesis tiene respuesta de sí o no y se convierte directamente en una lámina ("construcción que le vende al Estado convierte 2.5 veces más"). Además, cada hipótesis confirmada se vuelve una variable en el paso 7. **Por qué solo train:** si exploran test y ajustan decisiones mirando ahí, contaminan la evaluación con información que en la vida real no tendrían.

```sql
-- src/paso06_eda.py: el patrón se repite para cada hipótesis
SELECT sector, count(*) AS empresas, round(100 * avg(target), 1) AS tasa_pct
FROM train JOIN maestro USING (k)
WHERE seg NOT IN ('ESTADO', 'CORPORATIVO')
GROUP BY 1 ORDER BY tasa_pct DESC;
```

`avg(target)` es la tasa de conversión porque target vale 0 o 1: el promedio de ceros y unos es la proporción de unos. Es el truco más útil del EDA en clasificación.

**Qué salió (real).**

**H2 — ¿El rubro importa?** Sí, con 14 veces de diferencia entre extremos.

| Rubro | Empresas | Tasa de conversión |
| --- | --- | --- |
| Construcción | 23,853 | 12.7% |
| Agroexportación | 9,853 | 10.5% |
| Manufactura | 31,634 | 8.7% |
| Transporte y logística | 18,419 | 8.0% |
| Comercio mayorista | 37,067 | 7.2% |
| Agropecuario | 19,008 | 4.9% |
| Hoteles y restaurantes | 13,241 | 1.4% |
| Comercio minorista | 45,005 | 0.9% |
| Educación | 7,831 | 0.9% |

**H1 — ¿El plazo de cobro importa?** Es la relación más fuerte y más limpia del caso: la tasa sube en cada escalón.

| Plazo promedio de cobro | Empresas | Tasa |
| --- | --- | --- |
| Sin facturas B2B | 137,906 | 0.5% |
| Menos de 45 días | 4,340 | 3.7% |
| 45 a 59 días | 16,218 | 6.5% |
| 60 a 89 días | 81,516 | 12.7% |
| 90 días o más | 26,288 | 15.6% |

Esto confirma la lógica del producto: el factoring existe porque alguien te paga tarde. Sin facturas a otras empresas, casi nadie convierte (0.5%).

**H1b — Construcción que le vende al Estado:** 22.1% contra 8.8% de la que no le vende. Es el hallazgo más vendible: el Estado paga a 90 días o más y sus proveedores necesitan liquidez.

**H6 — ¿La regla actual de leads ignora el plazo?** Clientes Pyme:

| Lead | Plazo de cobro | Empresas | Tasa |
| --- | --- | --- | --- |
| No | Menos de 60 días o sin dato | 53,847 | 1.1% |
| No | 60 días o más | 27,149 | 9.1% |
| Sí | Menos de 60 días o sin dato | 1,490 | 18.3% |
| Sí | 60 días o más | 7,535 | 36.2% |

Aquí nace la forma B de crecer. Hay 27 mil Pyme **sin lead** que cobran a 60 días o más y convierten 9.1% incluso sin oferta: ocho veces más que el resto de las Pyme sin lead. Son el primer lugar donde buscar a quién habilitar.

**Tamaño del mercado (para la lámina 1):** comercio mayorista, manufactura y construcción concentran el mayor volumen facturado a grandes pagadores, y entre 80% y 90% de sus proveedores cobran a 60 días o más.

**Para cualquier otro caso.** El patrón es "tasa del target por grupo" para clasificación y "promedio del target por grupo" para regresión. Para una variable numérica, primero se corta en tramos (como el plazo). Reporten siempre cuántas empresas hay en cada grupo: una tasa de 40% con 12 empresas no significa nada.

**Pregunta de cierre:** el grupo "lead + plazo de 60 días o más" convierte 36%. ¿Eso prueba que el plazo causa la conversión? (No: muestra asociación. La causalidad la sostiene el argumento de negocio y, en la vida real, un piloto con grupo de control.)

## Paso 7 — Ingeniería de variables: de millones de eventos a una fila por empresa

**Qué hacemos.** Por cada fuente construimos una tabla con una fila por empresa (llave `k`) y columnas que resumen su comportamiento. Se guardan por separado en `features/`.

**Por qué.** El modelo necesita una fila por empresa, pero transacciones tiene 7.5 millones de eventos. La ingeniería de variables es cómo comprimimos esos eventos sin perder lo que predice. Es donde se gana el concurso: una buena variable de negocio sube el Gini más que cualquier ajuste del modelo. **Por qué por separado:** cada integrante puede trabajar su fuente en paralelo, y si una tabla falla no se rehacen las demás.

**La técnica depende del tipo de tabla.** Esta es la parte que les sirve para cualquier dato:

| Tipo de tabla | Ejemplo aquí | Técnica | Variables típicas |
| --- | --- | --- | --- |
| Atributos (una fila por entidad) | maestro | Se usan casi tal cual | segmento, rubro, zona, lead |
| Eventos (muchas filas por entidad) | transacciones, factoring | Agregar con `GROUP BY`: conteos, sumas, promedios, máximos, distintos, ventanas de tiempo, recencia | n\_facturas, vol\_ult3m, días desde la última |
| Panel ancho (una columna por mes) | perfil | Resumir la serie: nivel reciente, tendencia, volatilidad | saldo\_rec3, saldo\_tend, saldo\_cv |
| Relaciones (quién con quién) | pagador → proveedor | Atributos de la contraparte agregados | share\_grandes, vol\_estado |

### 7.1 La red pagador → proveedor (la fuente más rica)

```sql
CREATE TABLE fx AS            -- pagos de facturas + el segmento de QUIEN PAGA
SELECT x.*, p.seg AS seg_pag FROM 'limpio/transacciones.parquet' x
LEFT JOIN 'limpio/maestro.parquet' p ON x.ord = p.k
WHERE x.tipo IN ('PAGO_FACTURA', 'PAGO_PROVEEDORES_MASIVO');

-- 38% de dias_credito es nulo: lo imputo con el plazo TÍPICO de ese pagador
CREATE TABLE plazo_pagador AS SELECT ord, median(dias_credito) AS plazo_tipico FROM fx GROUP BY ord;

SELECT f.ben AS k,
  count(*) AS n_facturas, sum(monto_pen) AS vol_facturas, count(DISTINCT ord) AS n_pagadores,
  avg(coalesce(dias_credito, plazo_tipico)) AS plazo_prom,
  sum(monto_pen) FILTER (WHERE seg_pag IN ('GRANDE','CORPORATIVO','ESTADO')) / sum(monto_pen) AS share_grandes,
  sum(monto_pen) FILTER (WHERE fh >= DATE '2025-06-30' - INTERVAL 3 MONTH) AS vol_ult3m,
  date_diff('day', max(fh), DATE '2025-06-30') AS dias_desde_ult_factura
FROM fx LEFT JOIN plazo_pagador USING (ord) GROUP BY f.ben;
```

Por qué cada decisión:

- **El segmento del pagador** se trae con un join al maestro. Una factura de S/ 50 mil al Estado no es igual a una de S/ 50 mil a otra MIPE: la primera se cobra a 95 días.
- **Imputar el plazo con el del pagador** en vez de con la mediana global, porque los pagadores pagan de forma parecida a todos sus proveedores. Así se recupera una variable que el EDA mostró como la más fuerte.
- **`FILTER (WHERE …)`** calcula varias sumas condicionales en una sola pasada sobre los 7.5 millones de filas. Es la herramienta clave para agregar eventos.
- **Ventanas de 3, 6 y 12 meses:** lo reciente predice mejor que lo antiguo, y la diferencia entre ventanas mide si la empresa crece o se achica.
- **Recencia (`dias_desde_...`):** cuánto hace que pasó algo es casi siempre de las variables más predictivas.

### 7.2 Historial de factoring

Las mismas técnicas de agregación, con una línea que no puede faltar:

```sql
WHERE fecha_desembolso <= '2025-06-30' AND estado <> 'ANULADO'   -- la línea que evita la fuga (paso 5)
```

Se separa BCP de otros bancos: quien ya factoriza con la competencia es un lead caliente (H4).

### 7.3 Perfil: 12 meses en 3 números por variable

```python
rec = p[[f"{v}_m01", f"{v}_m02", f"{v}_m03"]].mean(axis=1)    # m01 = el más reciente
ant = p[[f"{v}_m10", f"{v}_m11", f"{v}_m12"]].mean(axis=1)
out[f"{v}_rec3"] = rec                                     # nivel actual
out[f"{v}_tend"] = (rec - ant) / (ant.abs() + 1)           # tendencia: + sube, - baja
out[f"{v}_cv"]   = p[meses].std(axis=1) / (p[meses].mean(axis=1).abs() + 1)   # volatilidad
```

Por qué no pasar las 96 columnas mensuales al modelo: están muy correlacionadas entre sí (el saldo de mayo se parece al de junio) y el modelo gastaría capacidad en ruido. Nivel, tendencia y volatilidad capturan lo que importa para H3 (una empresa con caja cayendo). El `+ 1` en el denominador evita dividir entre cero.

**Qué salió (real).** Todo el paso tomó 7 segundos y usó 1.5 GB como máximo:

```text
  f_maestro    filas= 400,000  k distintos= 400,000  variables=21
  f_red        filas= 183,061  k distintos= 183,061  variables=15   <- solo empresas que cobran facturas
  f_pagador    filas=  53,244  k distintos=  53,244  variables=2
  f_transf     filas= 399,474  k distintos= 399,474  variables=2
  f_factoring  filas=  36,995  k distintos=  36,995  variables=8    <- solo quienes hicieron factoring
  f_perfil     filas= 340,120  k distintos= 340,120  variables=28
```

En cada tabla, filas = k distintos: es la verificación de que el grano es "una fila por empresa". Que `f_red` tenga 183 mil filas y no 400 mil es esperado: solo las empresas que cobran facturas aparecen ahí.

**Para cualquier otro caso.** Ante una tabla de eventos, pregúntense: ¿cuántos? (conteo), ¿cuánto? (suma, promedio, máximo), ¿con quién? (distintos, atributos de la contraparte), ¿cuándo? (recencia, ventanas) y ¿cómo cambia? (crecimiento entre ventanas). Esas cinco preguntas generan el 80% de las variables útiles en casi cualquier caso: transacciones, llamadas, visitas, clics o retiros de efectivo.

**Pregunta de cierre:** ¿qué variables harían para el caso de 2024 (retiros de efectivo por agencia y día)? (Conteo y monto por día de la semana, por día del mes y en quincenas, promedio móvil de 7 y 28 días, feriados.)

## Paso 8 — Tabla de modelado y controles de calidad

**Qué hacemos.** Unimos todas las tablas de variables al maestro, decidimos qué hacer con los nulos, creamos variables derivadas y ejecutamos controles que detienen todo si algo está mal.

**Por qué.** Es el último punto donde un error es barato. Un join que duplicó filas o un target que se coló en test, si llega al modelo, produce un Gini falso que nadie cuestiona hasta que es tarde.

```python
# src/paso08_tabla.py (lo esencial)
df = con.execute("""SELECT * FROM f_maestro
    LEFT JOIN f_red USING (k) LEFT JOIN f_pagador USING (k) LEFT JOIN f_transf USING (k)
    LEFT JOIN f_factoring USING (k) LEFT JOIN f_perfil USING (k) LEFT JOIN target USING (k)""").df()

df[cero] = df[cero].fillna(0)        # nulo por "no hubo eventos" = 0
# nulo por "no se sabe" (score_buro, saldo de no clientes) se deja nulo

df["crec_facturas"] = (df.vol_ult6m - df.vol_prev6m) / (df.vol_prev6m + 1)
df["ya_factoriza_otro"] = (df.n_ops_otro > 0).astype("int8")

assert df.k.is_unique,                          "llave duplicada tras los joins"
assert len(df) == 400_000,                      "se perdieron o ganaron filas"
assert df[df.es_test == 1].target.sum() == 0,   "hay target en test: fuga"
```

**Las dos clases de nulo (decisión clave).**

| Tipo de nulo | Ejemplo | Qué significa | Qué hacer |
| --- | --- | --- | --- |
| No hubo eventos | `n_facturas` nulo tras el LEFT JOIN | La empresa no cobró ninguna factura: el valor real es 0 | Rellenar con 0 |
| No se sabe | `score_buro` nulo, saldo de un no cliente | Existe un valor, pero no lo conocemos | Dejar nulo: LightGBM aprende qué hacer con él |

Si rellenan "no se sabe" con 0, le dicen al modelo que el no cliente tiene saldo cero, que es falso. Y si rellenan con la media, borran la señal "esta empresa no es cliente", que sí predice.

**`assert` en vez de `print`.** Un `print` se pierde entre 200 líneas de salida a las 3 de la mañana. Un `assert` que falla detiene el pipeline con un mensaje claro. Son controles automáticos que corren cada vez.

**Qué salió (real).**

```text
tabla: 400,000 filas x 85 columnas | memoria 0.23 GB
columnas con mas nulos:
monto_oferta              0.967     <- solo los leads tienen oferta (esperado)
dias_desde_ult_op         0.908     <- solo quien hizo factoring (esperado)
tasa_prom_hist            0.908
uso_linea_pct_rec3        0.768     <- solo quien tiene línea de crédito (esperado)
columnas constantes (se descartan): []
```

Cada nulo alto tiene una explicación de negocio, y eso es lo que hay que verificar. Un 97% de nulos no es un problema si significa "no aplica". Sí lo sería si una columna que debería estar llena tuviera 50% de nulos: eso indicaría un join fallido.

De 8.9 millones de filas y 1.8 GB llegamos a 400 mil filas y 85 columnas en 0.23 GB. Desde aquí todo cabe cómodamente en pandas.

**Para cualquier otro caso.** Una fila por unidad de predicción; LEFT JOIN desde la tabla que define el universo; distinguir los dos tipos de nulo; y `assert` para la llave única, el número de filas y la ausencia de target en test.

**Pregunta de cierre:** ¿qué tipo de nulo es `dias_desde_ult_op` en una empresa que nunca hizo factoring, y por qué no se rellena con 0? (Es "no aplica". Rellenar con 0 diría "hizo factoring hoy", lo contrario de la verdad. Se deja nulo, o se rellena con un valor muy grande como 9999.)

## Paso 9 — Modelos de probabilidad: de lo simple a lo complejo

**Qué hacemos.** Entrenamos tres modelos en orden de complejidad, los validamos igual y nos quedamos con el que gana por un margen que valga la pena.

**Por qué tres y no directamente LightGBM.** El modelo simple es la vara de medir. Si LightGBM no le gana a una regla escrita a mano, algo está mal (fuga, variables rotas o un bug). Y si gana por muy poco, el modelo simple tiene una ventaja: se explica en una frase. Esa comparación es exactamente lo que impresiona al jurado técnico.

### 9.1 Cómo se valida: validación cruzada estratificada de 5 partes

Se divide train en 5 partes. Se entrena con 4 y se predice la quinta, 5 veces, hasta que cada empresa tiene una predicción hecha por un modelo que **no la vio**. Esas predicciones se llaman *out-of-fold* (OOF) y son la estimación honesta de cómo funcionará el modelo en test.

- **Por qué "estratificada":** con 5.8% de positivos, una partición al azar podría dejar una parte con 4% y otra con 7%. La estratificación asegura que cada parte tenga la misma tasa.
- **Por qué no un solo split 80/20:** con 5 partes se usa toda la data para validar, y la variación entre partes dice qué tan estable es el resultado.

### 9.2 Cómo se mide: Gini

El **AUC** es la probabilidad de que, tomando al azar una empresa que convirtió y una que no, el modelo le dé más puntaje a la que convirtió. Vale 0.5 al azar y 1.0 si el ordenamiento es perfecto. El **Gini = 2 × AUC − 1** reescala eso a 0 (azar) y 1 (perfecto); es como lo reporta la banca.

Por qué no accuracy: un modelo que dice "nadie convierte" acierta el 94.2% y no sirve para nada. Lo que el negocio necesita es **ordenar**: a quién llamar primero. El Gini mide justamente eso.

### 9.3 Los tres modelos

```python
# M0: regla de negocio escrita con las hipótesis del EDA (sin entrenar nada)
regla = 2*lead + (plazo_prom >= 60) + share_grandes + 0.2*log1p(vol_facturas) + (n_ops_bcp > 0)

# M1: regresión logística. No acepta nulos ni texto, por eso necesita:
#     imputar (mediana + indicador de nulo), escalar y one-hot de las categorías
pre = make_column_transformer(
    (make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), numericas),
    (OneHotEncoder(handle_unknown="ignore"), ["sector", "seg", "zona"]))
modelo = make_pipeline(pre, LogisticRegression(max_iter=2000, C=0.5))

# M2: LightGBM, con todas las variables tal cual (nulos y categorías incluidos)
params = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0)
for a, b in skf:
    m = lgb.train(params, lgb.Dataset(X.loc[a], y[a]), 3000,
                  valid_sets=[lgb.Dataset(X.loc[b], y[b])],
                  callbacks=[lgb.early_stopping(100)])
    oof[b] = m.predict(X.loc[b])
    pred_test += m.predict(X_test) / 5          # promedio de 5 modelos: más estable
```

**Qué hace cada parámetro de LightGBM (para responder si preguntan).**

| Parámetro | Qué controla | Por qué ese valor |
| --- | --- | --- |
| learning\_rate 0.03 | Cuánto corrige cada árbol nuevo | Bajo = más estable; el early stopping compensa con más árboles |
| num\_leaves 31 | Complejidad de cada árbol | Valor por defecto, suficiente para interacciones como sector × plazo |
| min\_child\_samples 100 | Mínimo de empresas por hoja | Evita reglas basadas en 5 empresas (sobreajuste) |
| feature\_fraction y bagging\_fraction 0.8 | Cada árbol ve 80% de columnas y filas | Diversidad entre árboles, menos sobreajuste |
| lambda\_l2 1.0 | Penalización de valores extremos en las hojas | Regularización suave |
| early\_stopping 100 | Para cuando 100 árboles seguidos no mejoran la validación | Decide solo cuántos árboles usar |

### 9.4 Qué salió (real)

| Modelo | Gini global | Gini en A (leads) | Gini en B (Pyme sin lead) | Tiempo |
| --- | --- | --- | --- | --- |
| M0 regla de negocio | 0.749 | 0.487 | 0.640 | 0 s |
| M1 regresión logística | 0.813 | 0.556 | 0.720 | 8 s |
| M2 LightGBM | 0.820 | 0.584 | 0.724 | 54 s |

```text
     árboles por fold: [270, 262, 209, 244, 237]
     Gini por fold: [0.82, 0.822, 0.815, 0.822, 0.821]     <- estable: no depende de la partición
```

**Cómo leerlo (con honestidad, que es lo que convence).**

1. **La regla de negocio ya da 0.749.** Las hipótesis del EDA eran buenas: la mayor parte de la señal es lógica de negocio.
2. **La logística sube a 0.813 y LightGBM solo agrega 0.007 más en global.** En data bien construida, la mayor parte del valor viene de las variables y no del algoritmo.
3. **Donde LightGBM sí gana es dentro de los leads (0.584 contra 0.556).** Ahí las empresas se parecen entre sí y hacen falta interacciones que la logística no ve. Ese es el argumento para usarlo: no "es más moderno", sino "ordena mejor justo donde la forma A lo necesita".
4. **El Gini en A es mucho menor que el global** porque los leads ya fueron filtrados: quedan empresas parecidas y ordenarlas es difícil. Reportar solo el global ocultaría esto.

**La tabla que entiende el negocio: lift por decil.** Se ordena a las empresas por probabilidad y se cortan en 10 grupos iguales:

| Decil | Tasa real | Lift | % de conversiones capturadas (acumulado) |
| --- | --- | --- | --- |
| 1 (10% más probable) | 36.7% | 6.3 | 63.0% |
| 2 | 10.9% | 1.9 | 81.6% |
| 3 | 5.5% | 0.9 | 91.0% |
| 4 | 2.5% | 0.4 | 95.2% |
| 5 a 10 | 0.0–1.2% | 0.0–0.2 | 100% |

En una frase para la lámina: **"Llamando solo al 10% que el modelo señala, capturamos el 63% de todas las conversiones"**. El lift de 6.3 significa que ese grupo convierte 6.3 veces más que el promedio.

**Para cualquier otro caso.** Siempre: (1) un baseline sin ML, (2) un modelo lineal, (3) un modelo de árboles; la misma validación para los tres; la métrica que pide el caso; y el resultado por grupo, no solo el global. Si el target es raro, validación estratificada; si el tiempo importa mucho (pronósticos), validación por tiempo: entrenar con el pasado y validar con el período siguiente.

**Pregunta de cierre:** el jurado pregunta por qué usaron LightGBM si la logística da casi lo mismo. ¿Qué responden? (Que en global es casi igual, pero dentro de los leads ordena 0.03 mejor, y ahí es donde se decide a quién llamar primero. Y que la logística la usan para validar que el resultado tiene sentido.)

## Paso 10 — Modelo de monto y valor esperado

**Qué hacemos.** Dos regresiones distintas para dos preguntas distintas, y comparamos contra una alternativa ingenua.

**Por qué dos modelos.** "¿Cuánto financiará esta empresa?" tiene dos lecturas. (R1) El monto esperado contando que la mayoría no convierte: es lo que mide el R² del concurso, sobre todas las empresas. (R2) Cuánto financiará **si** convierte: es lo que necesita el negocio para calcular el valor de una visita, valor esperado = probabilidad × monto si convierte. Mezclarlas en un solo modelo da un número que no sirve bien para ninguna de las dos. A esta estructura se la llama **modelo de dos partes** (*hurdle*).

**Por qué en logaritmo (`log1p`).** Miren la distribución real del monto entre los que convierten:

```text
mean       1,128,654
50%          321,116     <- la mitad financia menos de S/ 321 mil
90%        2,949,630
99%       11,325,040
max       42,336,373     <- una empresa financia 130 veces la mediana
```

Con esa cola larga, un modelo en soles se dedicaría a acertar las 50 empresas gigantes e ignoraría a las Pyme. En logaritmo, equivocarse por el doble pesa igual en una empresa chica que en una grande. Se usa `log1p(x) = log(1 + x)` porque log(0) no existe y aquí hay muchos ceros; para volver a soles se usa `expm1`.

```python
# src/paso10_monto.py (lo esencial)
oof_all, pred_all = cv_reg(train,      np.log1p(train.monto_target))       # R1: todas, con ceros
oof_pos, pred_pos = cv_reg(positivos,  np.log1p(positivos.monto_target))   # R2: solo los que convierten
monto_esperado     = np.expm1(pred_all).clip(0)
monto_si_convierte = np.expm1(pred_pos)
```

**Qué salió (real).**

| Modelo | R² en log(1 + monto), validación cruzada |
| --- | --- |
| R1: monto de todas las empresas (para la métrica) | 0.291 |
| R2: monto si convierte (para el valor esperado) | 0.810 |
| Ingenuo: probabilidad × monto promedio | −8.539 |

**Cómo leerlo.**

- **R2 = 0.81:** dado que una empresa convierte, su monto es muy predecible, porque depende de cuánto factura y a quién. El monto no es el problema difícil.
- **R1 = 0.29:** es bajo porque hereda la dificultad de adivinar quién convierte (el paso 9). No es un mal modelo: la pregunta es más difícil.
- **El ingenuo da −8.5:** un R² negativo significa que es peor que predecir el promedio para todos. Multiplicar una probabilidad por un promedio en soles produce números intermedios para todos, y en escala logarítmica eso es un desastre. Es el error típico de quien arma el valor esperado sin pensar en cómo se mide.

**Para cualquier otro caso.** Si el target numérico tiene muchos ceros, usen dos partes (¿ocurre? × ¿cuánto si ocurre?). Si tiene cola larga, transformen con log. Y siempre optimicen lo mismo que mide la métrica: si el concurso mide R² en log, entrenen en log.

**Pregunta de cierre:** ¿por qué para priorizar visitas usamos prob × monto\_si\_convierte y no el monto\_esperado de R1? (Porque R1 está optimizado para el error en log y subestima los montos grandes. Para decidir en soles se necesita la esperanza en soles: probabilidad por monto si convierte.)

## Paso 11 — Habilitar y priorizar con capacidad limitada

**Qué hacemos.** Convertimos probabilidades y montos en una decisión: a quién darle lead (forma B) y a quién visitar, respetando la capacidad (2.5% de las empresas de evaluación) y los cupos por zona.

**Por qué este paso decide la final.** Un modelo con Gini de 0.82 no le dice al banco qué hacer el lunes. La lista de visitas sí, y es lo que evalúa el jurado de negocio. Aquí entra la parte que la lámina deja a cada equipo: **su criterio de "cliente más valioso"**.

```python
# src/paso11_priorizar.py (lo esencial)
te["valor_soles"] = te.prob * te.monto_si_convierte * 0.018 - 150     # ingreso esperado - costo de la visita
te["score"] = te.valor_soles            # o te.prob, según el criterio elegido

# 1) HABILITAR: entre empresas SIN lead, las de mayor score, hasta duplicar la base de leads
n_leads = te.lead.sum()
te.loc[te[(te.lead == 0) & elegible].nlargest(n_leads, "score").index, "habilitar"] = 1

# 2) PRIORIZAR: solo quien tiene lead (actual o habilitado), por zona, hasta llenar el cupo
K = round(0.025 * len(te))
cand = te[((te.lead == 1) | (te.habilitar == 1)) & elegible & (te.zona != "DESCONOCIDA")]
for zona, p in CUPOS.items():
    te.loc[cand[cand.zona == zona].nlargest(int(K * p), "score").index, "priorizado"] = 1
```

**Por qué ordenar y cortar es óptimo aquí.** Cada visita cuesta lo mismo y los cupos de cada zona son independientes. Entonces, para maximizar la suma de valores en una zona con N cupos, basta tomar las N empresas de mayor valor: cualquier cambio reemplaza a una empresa por otra de menor valor. Este método se llama **voraz** (*greedy*). Si el jurado agrega restricciones que se cruzan (máximo por ejecutivo y por canal a la vez, o visitas con costos distintos), el orden ya no basta y se usa **programación lineal entera**, con la librería `pulp`.

**Por qué se excluye la zona desconocida.** Si una empresa sin zona se asigna a Lima "por defecto" y en realidad es del Norte, rompe el cupo del Norte sin que lo vean. Con la solución de referencia pasó justo eso con 3 empresas.

**Qué salió (real): el mismo modelo con tres criterios de valor.**

| Criterio | Visitas: leads A / habilitados | Conversiones (tasa) | Pyme convertidas | Monto financiado | Valor neto de referencia |
| --- | --- | --- | --- | --- | --- |
| Soles: máximo valor neto, todo el universo | 162 / 2,832 | 1,400 (46.8%) | 83 | S/ 5,052 M | S/ 90.5 M |
| Pyme: máximo valor, solo Pyme | 1,691 / 1,303 | 1,341 (44.8%) | 1,341 | S/ 498 M | S/ 8.5 M |
| Conversiones: máxima probabilidad, solo Pyme | 1,603 / 1,391 | 1,434 (47.9%) | 1,434 | S/ 468 M | S/ 8.0 M |

Las tres convierten entre 45% y 48% de las visitas, frente a una base de 5.8%: el modelo funciona con cualquier criterio. Lo que cambia es **qué tipo de crecimiento** compran:

- **Soles** casi no visita a los leads actuales (162 de 2,994) y habilita medianas y grandes: 10 veces más monto, pero 16 veces menos Pyme. Contradice el foco Pyme de la lámina.
- **Pyme** reparte las visitas casi mitad y mitad entre las formas A y B. De sus 3,955 habilitados, 2,123 son clientes Pyme sin lead y 1,832 son Pyme que todavía no son clientes del BCP: es habilitación y adquisición de clientes a la vez.
- **Conversiones** maximiza el número de Pyme que entran al producto (1,434), sacrificando un 6% de monto frente al criterio Pyme.

**Cómo elegir y defenderlo.** No hay criterio correcto: hay uno coherente con un objetivo. Una defensa sólida tiene tres partes: (1) el objetivo ("crecer la base Pyme, como dice la lámina"), (2) el costo de esa elección con números ("renunciamos a S/ 4,500 M de monto de medianas") y (3) una mitigación ("proponemos un canal aparte para medianas"). Para correr cada criterio: `python src/paso11_priorizar.py pyme` (o `soles`, o `conversiones`).

**Para cualquier otro caso.** Todo problema de "a quién atender con recursos limitados" se resuelve igual: un puntaje de valor por unidad, las restricciones escritas explícitamente y ordenar y cortar (o programación lineal si las restricciones se cruzan). En 2024 el recurso eran los camiones de transporte de caudales; en 2025, las llamadas por canal.

**Pregunta de cierre:** si la capacidad baja a la mitad, ¿hay que reentrenar el modelo? (No: el modelo da el orden y la capacidad solo mueve el corte. Se vuelve a correr el paso 11, que tarda 1 segundo.)

## Paso 12 — Explicar el modelo y convertirlo en historia

**Qué hacemos.** Tres explicaciones a tres niveles: qué fuente de datos aportó más, qué variables mueven el puntaje y por qué una empresa concreta salió alta.

**Por qué.** El jurado de negocio no compra un número que no entiende. "El modelo dice 61%" no convence; "esta Pyme agroexportadora ya hace factoring con otro banco, cobra a 121 días y factura S/ 4.4 millones" sí. Además, explicar es la última prueba contra la fuga: si la variable principal no tiene sentido de negocio, hay un problema.

### 12.1 ¿Qué fuente de datos aportó más? (importancia agrupada)

```text
f_factoring    49.2%     historial de factoring antes del corte
f_red          33.5%     red pagador -> proveedor (transacciones)
f_maestro      10.4%
f_perfil        6.7%
```

Lectura: el historial de factoring y la red de pagos explican el 83% del modelo. El perfil de saldos, con 116 columnas, aporta solo el 6.7%. Moraleja para el día del concurso: la tabla con más columnas no es la más valiosa, y el tiempo se invierte donde está la lógica del producto.

### 12.2 ¿Qué mueve el puntaje? (SHAP)

La importancia por "ganancia" dice cuánto se usó una variable, no cuánto mueve la predicción. Para eso están los **valores SHAP**: para cada empresa reparten la predicción entre las variables, de modo que la suma de los aportes da el puntaje final. LightGBM los calcula sin instalar nada:

```python
contrib = modelo.predict(X, pred_contrib=True)   # una columna por variable + la base, en log-odds
```

```text
SHAP medio absoluto (cuanto mueve cada variable el score):
vol_facturas                0.321
plazo_max                   0.281
sector                      0.246
seg                         0.226
ticket_prom                 0.211
dias_desde_ult_op           0.190
```

Con SHAP, las variables de la red y del negocio (volumen facturado, plazo, rubro, segmento) quedan arriba. Por ganancia aparecían primero las de historial de factoring. La diferencia es que el historial es decisivo para las 37 mil empresas que lo tienen, mientras que el volumen y el plazo mueven el puntaje de todas.

### 12.3 ¿Por qué esta empresa? (explicación individual)

```text
Empresa k=686153: PYME, sin lead, nunca hizo factoring con el BCP, sector AGROEXPORTACION
probabilidad 61% | convirtió realmente: sí
   dias_desde_ult_op     22         +0.91   <- hizo factoring con OTRO banco hace 22 días
   tasa_prom_hist        1.36       +0.68
   vol_facturas          4,368,153  +0.40   <- factura S/ 4.4 M a otras empresas
   sector                AGROEXP.   +0.36
   plazo_max             121        +0.30   <- le pagan hasta a 121 días
   ticket_prom           104,004    +0.27
```

Esta es la lámina de la forma B en un solo ejemplo: una Pyme sin lead, invisible para la regla actual, que ya usa factoring en la competencia, cobra a 4 meses y está en un rubro estacional. El modelo la encontró y efectivamente convirtió. Los aportes están en log-odds: lo que importa es el signo y el tamaño relativo.

### 12.4 ¿En qué rubros confío más en el modelo?

El Gini por rubro va de 0.73 (agroexportación) a 0.81 (hoteles y restaurantes). Que sea parejo es bueno: el modelo no depende de un solo rubro. Agroexportación tiene el Gini más bajo y una de las tasas más altas: su estacionalidad hace la predicción más difícil. Eso es una recomendación de mejora para la diapositiva de siguientes pasos.

**Para cualquier otro caso.** Siempre las tres capas: por fuente (¿dónde invertir tiempo?), por variable con SHAP (¿qué lógica aprendió?) y por caso individual (¿cómo se lo cuento a un gerente?). Si la variable número 1 no tiene una explicación de negocio en una frase, revisen fugas antes de presentar.

**Pregunta de cierre:** si la variable más importante fuera `codigo_interno`, ¿qué harían? (Es un identificador sin significado de negocio: probablemente está correlacionado con la fecha de alta o con cómo se armó la data. Se elimina y se investiga por qué predecía.)

## Paso 13 — Pipeline reproducible y trabajo en equipo con Git

**Qué hacemos.** Un script que ejecuta todos los pasos en orden y se detiene si uno falla, más un flujo de Git para que 5 personas trabajen sin pisarse.

**Por qué.** El jurado técnico revisa el código. "Clonen el repo y corran un comando" demuestra un nivel profesional que un notebook con celdas en desorden no muestra. Además, a las 4 de la mañana alguien cambia la limpieza y hay que regenerar todo sin pensar.

```python
# src/run_all.py
PASOS = ["paso04_limpiar.py", "paso05_target.py", "paso07_features.py", "paso08_tabla.py",
         "paso09_modelo.py", "paso10_monto.py", "paso11_priorizar.py pyme"]
for p in PASOS:
    subprocess.run([sys.executable] + p.split(), check=True)   # check=True: si un paso falla, se detiene
```

**Qué salió (real), desde cero:**

```text
<<< paso04_limpiar.py: 49 s
<<< paso05_target.py: 2 s
<<< paso07_features.py: 7 s
<<< paso08_tabla.py: 3 s
<<< paso09_modelo.py: 65 s
<<< paso10_monto.py: 47 s
<<< paso11_priorizar.py pyme: 1 s
TOTAL: 2.9 min -> salidas/entrega_pyme.csv
```

De 8.9 millones de filas crudas a la entrega en menos de 3 minutos. En el concurso eso significa que pueden probar una idea nueva y ver su efecto en el Gini 15 o 20 veces durante la noche.

**Por qué un archivo por paso y no un notebook.** Los notebooks permiten ejecutar celdas en desorden y guardar resultados que ya no corresponden al código. Además, en Git sus cambios son ilegibles y los conflictos de merge son casi imposibles de resolver. Úselos solo para explorar y pasen a `src/` lo que funcione.

**Flujo de Git del equipo (ensáyenlo antes del 13).**

```bash
git checkout -b de/limpieza            # una rama por persona y tema
# ... trabajar en SU archivo, por ejemplo src/paso04_limpiar.py ...
git add src/paso04_limpiar.py
git commit -m "limpieza: dedup maestro por fecha_actualizacion"
git push -u origin de/limpieza
# en GitHub: Pull Request hacia main -> el líder revisa y hace merge
git checkout main && git pull           # todos actualizan main después de cada merge
```

| Regla | Por qué |
| --- | --- |
| Cada persona edita solo sus archivos `pasoNN` | Si nadie toca el mismo archivo, no hay conflictos de merge |
| Los pasos se comunican por archivos Parquet con contrato fijo (llave `k`, una fila por empresa) | Cada uno trabaja en paralelo sin esperar al otro |
| `data/`, `limpio/`, `features/` y `salidas/` en `.gitignore` | Son grandes y se regeneran con `run_all.py` |
| Commits pequeños con mensajes que dicen el porqué | El README y la defensa técnica salen del historial |
| `main` siempre corre de punta a punta | Siempre hay una entrega válida que enviar |

**Para cualquier otro caso.** Un pipeline sirve para cualquier problema: pasos numerados, entrada y salida en disco, controles con `assert` y un comando que corre todo. Es lo que en la industria se llama un pipeline de datos y se ve igual en un banco, un retail o una startup.

**Pregunta de cierre:** el Data Scientist cambia una variable del paso 7. ¿Qué pasos hay que volver a correr? (7, 8, 9, 10 y 11. Los pasos 4 y 5 no cambian, porque sus salidas en disco siguen siendo válidas.)

## Cómo adaptarse a cualquier dato o caso: receta universal

Los 13 pasos no son "del caso factoring": son el orden de trabajo de cualquier problema de datos. Lo que cambia de un caso a otro son las respuestas a 8 preguntas. Contéstenlas por escrito en los primeros 40 minutos del concurso y el resto de la guía se aplica casi igual.

**Las 8 preguntas, ante cualquier data.**

1. **¿Qué decisión de negocio se va a tomar?** (a quién llamar, cuánto efectivo enviar, por qué canal contactar)
2. **¿Cuál es la unidad de predicción?** (empresa, agencia × día, cliente × canal). Esa será la fila de la tabla de modelado.
3. **¿Cuál es el target y en qué ventana de tiempo?**
4. **¿Cuál es la fecha de corte?** Todo lo que sea posterior es fuga.
5. **¿Cuál es el grano y la llave de cada tabla?** ¿Cómo llego desde cada tabla a la unidad de predicción?
6. **¿Cuánto pesa?** (paso 1: pandas o DuckDB)
7. **¿Qué métrica evalúa el jurado?** Optimicen exactamente esa.
8. **¿Qué restricción tiene la decisión?** (capacidad, presupuesto, cupos)

**Según el tipo de problema.**

| Si el enunciado dice… | Tipo | Modelo | Métrica | Validación | Ejemplo DataFest |
| --- | --- | --- | --- | --- | --- |
| "probabilidad de que…", "quién va a…" | Clasificación binaria | Logística + LightGBM | Gini / AUC, lift | K-fold estratificado | 2026: factoring |
| "cuánto…" (por entidad) | Regresión | LightGBM en log si hay cola larga; dos partes si hay muchos ceros | R², RMSE, MAE | K-fold | 2026: monto |
| "cuánto mañana / la próxima semana" | Pronóstico de series de tiempo | Promedio estacional como baseline; LightGBM con rezagos y calendario | MAPE, WAPE, RMSE | **Por tiempo**: entrenar con el pasado, validar con el período siguiente | 2024: efectivo en agencias |
| "cuál canal / producto / momento" | Multiclase o un modelo por opción | Un modelo binario por canal y elegir el de mayor valor esperado | AUC por clase, valor capturado | K-fold estratificado | 2025: mejor canal |
| "con capacidad limitada", "minimizar costo" | Optimización | Ordenar y cortar; programación lineal (`pulp`) si las restricciones se cruzan | Valor capturado vs óptimo | Sobre las predicciones validadas | 2024 y 2026 |
| "segmentar", "grupos de clientes" (sin target) | Agrupamiento | K-means sobre variables estandarizadas | Que los grupos se puedan explicar | Estabilidad con otra semilla | Apoyo al storytelling |

**El pronóstico es el caso que más cambia la guía.** En 2024 la unidad era agencia × día. Las variables son rezagos (el retiro de hace 1, 7 y 28 días), promedios móviles, día de la semana, día del mes, quincena, feriados y gratificaciones. La validación nunca puede ser aleatoria: si se entrena con el miércoles para predecir el martes, eso es fuga. Siempre se entrena con el pasado y se valida con el período siguiente.

**Según el tipo de dato.**

| Dato | Qué hacer |
| --- | --- |
| Atributos (una fila por entidad) | Usar casi tal cual; normalizar categorías |
| Eventos (muchas filas por entidad) | Agregar: conteo, suma, promedio, máximo, distintos, recencia y ventanas (paso 7) |
| Panel ancho (una columna por mes) | Resumir: nivel reciente, tendencia y volatilidad |
| Series de tiempo | Rezagos, promedios móviles, calendario; validación por tiempo |
| Relaciones (quién con quién) | Atributos de la contraparte agregados; número de contrapartes |
| Texto libre | Normalizar (mayúsculas, tildes, espacios); buscar palabras clave; usar un código estructurado si existe (como el CIIU) |
| Categorías con miles de valores | Tipo `category` de LightGBM, o agrupar las raras en "OTROS" |
| Fechas | Varios formatos con `try_strptime`; derivar antigüedad, mes y día de la semana |
| Montos | Una sola moneda; centinelas a nulo; log si hay cola larga |

**Según el tamaño:** la tabla del paso 1. Hasta 1/4 de la RAM, pandas; más, DuckDB hasta llegar a una fila por unidad de predicción. Esa tabla final casi siempre cabe en pandas.

**Lo que nunca cambia.** Medir antes de abrir. Perfilar antes de limpiar. Llaves y grano antes de unir. Fecha de corte antes de crear variables. Baseline antes de modelos complejos. Métrica del jurado, no la favorita. Resultados por grupo, no solo el global. Y un comando que corra todo.

**Pregunta de cierre del curso:** el 13 les entregan datos de llamadas de un call center y les piden "reducir el abandono". Respondan las 8 preguntas antes de leer cualquier columna. (Decisión: a quién retener. Unidad: cliente. Target: abandona en los 3 meses siguientes. Corte: la fecha de la data. Grano: llamada, cliente. Tamaño: medir. Métrica: preguntar. Restricción: capacidad del equipo de retención.)

## Glosario rápido

| Término | Qué es, en una frase |
| --- | --- |
| Grano | Lo que representa una fila de una tabla (una empresa, un pago, un mes) |
| Llave | Columna (o columnas) que identifica cada fila y permite unir tablas |
| Join izquierdo (LEFT JOIN) | Une conservando todas las filas de la tabla de la izquierda, aunque no tengan pareja |
| Función de ventana | Cálculo dentro de grupos sin colapsarlos, como `row_number() OVER (PARTITION BY …)` |
| Centinela | Valor falso que significa "no se sabe" (-999, 99999999, 1900-01-01) |
| Parquet | Formato de archivo columnar, comprimido y con tipos; mucho más rápido que CSV |
| Fecha de corte | El "hoy" del modelo: nada posterior puede usarse para crear variables |
| Fuga de información | Usar información que no existiría al momento de predecir; infla la validación |
| Target | Lo que se quiere predecir |
| Desbalanceado | Una clase es mucho más rara que la otra (aquí 5.8% contra 94.2%) |
| Feature (variable) | Columna que el modelo usa para predecir |
| Recencia | Tiempo desde el último evento |
| Baseline | Modelo o regla más simple posible, que sirve de vara de medir |
| Validación cruzada (K-fold) | Entrenar K veces dejando fuera una parte distinta cada vez |
| Estratificada | Validación cruzada que mantiene la misma proporción del target en cada parte |
| OOF (out-of-fold) | Predicción hecha por un modelo que no vio esa fila |
| AUC | Probabilidad de que el modelo puntúe más alto a un positivo que a un negativo elegidos al azar |
| Gini | 2 × AUC − 1; 0 es azar y 1 es orden perfecto |
| Lift | Cuántas veces convierte un grupo comparado con el promedio |
| R² | Qué parte de la variación del target explica el modelo; negativo = peor que el promedio |
| log1p / expm1 | log(1 + x) y su inversa; comprimen colas largas y aceptan ceros |
| Modelo de dos partes (hurdle) | Probabilidad de que ocurra × cuánto si ocurre |
| Boosting | Árboles en secuencia, cada uno corrige los errores del anterior (LightGBM, XGBoost) |
| Early stopping | Dejar de agregar árboles cuando la validación deja de mejorar |
| Sobreajuste | El modelo memoriza train y falla con datos nuevos |
| SHAP | Reparto de la predicción de cada fila entre sus variables |
| Voraz (greedy) | Ordenar por valor y tomar los mejores hasta llenar la capacidad |
| Programación lineal entera | Optimización con restricciones cruzadas y decisiones sí o no (librería `pulp`) |
| Pipeline | Secuencia de pasos automatizada y reproducible con un comando |
