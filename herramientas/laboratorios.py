"""Genera la versión para estudiantes de los laboratorios a partir del repo privado.

Las fuentes con solución viven en `ml-unison/soluciones` (privado), con esta estructura:

    soluciones/
      unidades/01-introduccion/u01-lab01-....ipynb   (o .py en formato percent)

Cada fuente usa el formato de otter-grader (https://otter-grader.readthedocs.io):
celdas raw `# ASSIGNMENT CONFIG`, `# BEGIN QUESTION`, `# BEGIN SOLUTION`/`# END SOLUTION`
alrededor de las celdas de solución, `# BEGIN TESTS`/`# END TESTS`, y
y un bloque "BEGIN TEST CONFIG" con `hidden: true` para las pruebas ocultas.

Este script, para cada fuente:
  1. fuerza la configuración del curso (pruebas en archivos; en Colab se descargan de este repo),
  2. ejecuta `otter assign`, que verifica que la solución pase todas las pruebas,
  3. copia la libreta del estudiante y sus pruebas públicas a unidades/<unidad>/laboratorios/,
  4. agrega la celda inicial del curso (RAMA, instalación de rdp),
  5. verifica que ninguna línea de solución llegue a la versión del estudiante,
  6. guarda el paquete del autocalificador en <soluciones>/autocalificador/ (para `otter grade`).

Las respuestas y las pruebas ocultas nunca se copian a este repositorio.

Uso:
    uv run python herramientas/laboratorios.py --soluciones ../soluciones
    uv run python herramientas/laboratorios.py --soluciones ../soluciones u01-lab01
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import jupytext
import nbformat
import nbformat.validator
import yaml

RAIZ = Path(__file__).resolve().parents[1]
REPO = "ml-unison/ml-unison.github.io"
OTTER = [sys.executable, "-m", "otter"]  # otter-grader del grupo dev, con las dependencias del curso

CELDA_INICIAL = '''import os, sys

RAMA = "{rama}"  # rama del repositorio del curso de la que se toman paquete, datos y pruebas
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    %pip install -q otter-grader "rdp @ git+https://github.com/{repo}@{{RAMA}}#subdirectory=paquete"'''

# otter-grader solo sabe descargar pruebas por URL en JupyterLite; en Colab las bajamos nosotros.
CELDA_OTTER = '''# Verificador de respuestas (otter-grader). Usa las pruebas públicas del laboratorio:
# la copia local si existe; si no (en Colab), las descarga del repositorio del curso.
import pathlib, urllib.request
import otter

PRUEBAS = pathlib.Path("tests") / "{lab}"
if not PRUEBAS.exists():
    PRUEBAS.mkdir(parents=True)
    for _p in {archivos}:
        urllib.request.urlretrieve(
            f"https://raw.githubusercontent.com/{repo}/{{RAMA}}/unidades/{unidad}/laboratorios/tests/{lab}/{{_p}}",
            PRUEBAS / _p)
grader = otter.Notebook(tests_dir=str(PRUEBAS))'''


def rama_actual() -> str:
    """La rama que usan las libretas de ejemplo (fuente de verdad: herramientas/libretas.py rama)."""
    for py in sorted((RAIZ / "unidades").rglob("*.py")):
        m = re.search(r'^RAMA = "([^"]+)"', py.read_text(encoding="utf-8"), re.M)
        if m:
            return m.group(1)
    return "main"


def leer(ruta: Path) -> nbformat.NotebookNode:
    if ruta.suffix == ".py":
        return jupytext.read(ruta, fmt="py:percent")
    return nbformat.read(ruta, as_version=4)


def configurar(nb: nbformat.NotebookNode) -> None:
    celda = next((c for c in nb.cells if c.cell_type == "raw" and c.source.lstrip().startswith("# ASSIGNMENT CONFIG")), None)
    if celda is None:
        raise ValueError("falta la celda raw '# ASSIGNMENT CONFIG'")
    cuerpo = celda.source.split("\n", 1)[1] if "\n" in celda.source else ""
    config = yaml.safe_load(cuerpo) or {}
    # generate=True arma el paquete del autocalificador, que otter usa para correr las pruebas
    # contra la solución; strip_solutions_from_all_cells aplica las marcas "# SOLUTION" en cualquier celda.
    config.update({"runs_on": "default", "export_cell": False, "generate": True, "solutions_pdf": False,
                   "strip_solutions_from_all_cells": True, "run_tests": True})
    config.setdefault("check_all_cell", True)
    config["tests"] = {**config.get("tests", {}), "files": True}
    config["tests"].pop("url_prefix", None)
    celda.source = "# ASSIGNMENT CONFIG\n" + yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}


def lineas_de_solucion(nb: nbformat.NotebookNode) -> set[str]:
    """Líneas de código que son solución (marcadas con # SOLUTION o dentro de BEGIN/END SOLUTION)."""
    lineas, enunciado, dentro, prompt = set(), set(), False, False
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        for ln in c.source.splitlines():
            t = ln.strip()
            # las líneas del enunciado (bloques BEGIN/END PROMPT) sí las ve el estudiante
            if t.endswith("# BEGIN PROMPT"):
                prompt = True
                continue
            if t.endswith("# END PROMPT"):
                prompt = False
                continue
            if prompt:
                enunciado.add(t)
                continue
            if t.startswith("# BEGIN SOLUTION"):
                dentro = True
                continue
            if t.startswith("# END SOLUTION"):
                dentro = False
                continue
            if t.endswith("# SOLUTION"):
                lineas.add(t.removesuffix("# SOLUTION").strip())
            elif dentro and len(t) > 12:
                lineas.add(t)
    return {ln for ln in lineas if len(ln) > 12 and ln not in enunciado}


def verificar_sin_soluciones(fuente: nbformat.NotebookNode, alumno: nbformat.NotebookNode, nombre: str) -> None:
    codigo = "\n".join(c.source for c in alumno.cells if c.cell_type == "code")
    filtradas = [ln for ln in lineas_de_solucion(fuente) if ln in codigo]
    if filtradas:
        raise RuntimeError(f"{nombre}: la versión del estudiante contiene soluciones:\n  " + "\n  ".join(filtradas[:5]))


def procesar(fuente: Path, base_soluciones: Path, rama: str) -> Path:
    relativa = fuente.parent.relative_to(base_soluciones / "unidades")  # p. ej. 01-introduccion
    destino = RAIZ / "unidades" / relativa / "laboratorios"

    nb = leer(fuente)
    configurar(nb)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        entrada = tmp / f"{fuente.stem}.ipynb"
        nbformat.write(nb, entrada)
        # otter assign ejecuta la solución y verifica que pase todas las pruebas
        entorno = {**os.environ, "RDP_DATOS": str(RAIZ / "datos"), "RDP_RAMA": rama}
        r = subprocess.run([*OTTER, "assign", str(entrada), str(tmp / "dist"), "-v"],
                           capture_output=True, text=True, env=entorno)
        estudiante = tmp / "dist" / "student" / entrada.name
        if r.returncode != 0 or not estudiante.exists():
            print(r.stdout[-3000:], r.stderr[-3000:], file=sys.stderr)
            raise RuntimeError(f"otter assign falló para {fuente.name} (¿la solución no pasa sus pruebas?)")

        alumno = nbformat.read(estudiante, as_version=4)
        nbformat.validator.normalize(alumno)
        texto = json.dumps(alumno)
        if "# BEGIN SOLUTION" in texto or "hidden: true" in texto:
            raise RuntimeError(f"{fuente.name}: la versión del estudiante conserva marcas de solución")
        verificar_sin_soluciones(nb, alumno, fuente.name)
        archivos = sorted(t.name for t in (tmp / "dist" / "student" / "tests").glob("*.py"))
        for c in alumno.cells:
            if c.cell_type == "code" and "otter.Notebook(" in c.source:
                c.source = CELDA_OTTER.format(lab=fuente.stem, archivos=archivos, repo=REPO, unidad=relativa.as_posix())
        alumno.cells = [c for c in alumno.cells if not (c.cell_type == "code" and c.source.strip().startswith("%pip install") and "otter" in c.source)]
        inicial = nbformat.v4.new_code_cell(CELDA_INICIAL.format(rama=rama, repo=REPO))
        alumno.cells.insert(0, inicial)
        # el encabezado del curso va primero, antes de las celdas de instalación
        k = next((i for i, c in enumerate(alumno.cells) if c.cell_type == "markdown" and "Universidad de Sonora" in c.source), None)
        if k:
            alumno.cells.insert(0, alumno.cells.pop(k))
        for c in alumno.cells:
            if c.cell_type == "code":
                c.outputs, c.execution_count = [], None
        alumno.metadata["colab"] = {"provenance": [], "toc_visible": True}

        destino.mkdir(parents=True, exist_ok=True)
        nbformat.write(alumno, destino / entrada.name)
        # Solo las pruebas públicas (las ocultas se quedan en el autograder)
        pruebas = destino / "tests" / fuente.stem
        shutil.rmtree(pruebas, ignore_errors=True)
        pruebas.mkdir(parents=True)
        for t in (tmp / "dist" / "student" / "tests").glob("*.py"):
            shutil.copy(t, pruebas / t.name)
        # El paquete del autocalificador (con pruebas ocultas) se queda en el repo privado
        paquete = next((tmp / "dist" / "autograder").glob("*.zip"), None)
        if paquete:
            (base_soluciones / "autocalificador").mkdir(exist_ok=True)
            shutil.copy(paquete, base_soluciones / "autocalificador" / f"{fuente.stem}.zip")
    return destino / entrada.name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--soluciones", type=Path, default=RAIZ.parent / "soluciones",
                    help="ruta al clon local de ml-unison/soluciones")
    ap.add_argument("filtro", nargs="*", help="solo laboratorios cuyo nombre contenga alguno de estos textos")
    args = ap.parse_args(argv)

    base = args.soluciones.resolve()
    if not (base / "unidades").is_dir():
        print(f"No encuentro {base}/unidades. Clona ml-unison/soluciones o usa --soluciones.", file=sys.stderr)
        return 1
    rama = rama_actual()
    fuentes = sorted(p for p in (base / "unidades").rglob("*-lab*") if p.suffix in {".py", ".ipynb"})
    if args.filtro:
        fuentes = [p for p in fuentes if any(f in p.name for f in args.filtro)]
    for f in fuentes:
        print(f"✓ {procesar(f, base, rama).relative_to(RAIZ)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
