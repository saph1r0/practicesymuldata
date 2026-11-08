"""PASO 13 - Todo de punta a punta con UN comando: python src/run_all.py [criterio]
Si un paso falla, se detiene (check=True): nunca se entrega con un paso intermedio roto."""
import subprocess
import sys
import time
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
criterio = sys.argv[1] if len(sys.argv) > 1 else "pyme"
PASOS = ["paso04_limpiar.py", "paso05_target.py", "paso07_features.py", "paso08_tabla.py",
         "paso09_modelo.py", "paso10_monto.py", f"paso11_priorizar.py {criterio}"]
t0 = time.time()
for p in PASOS:
    t = time.time()
    print(f"\n>>> {p}", flush=True)
    subprocess.run([sys.executable] + [os.path.join(AQUI, p.split()[0])] + p.split()[1:], check=True, cwd=AQUI)
    print(f"<<< {p}: {time.time() - t:.0f} s", flush=True)
print(f"\nTOTAL: {(time.time() - t0) / 60:.1f} min -> salidas/entrega_{criterio}.csv")
