"""Genera la versión para estudiantes de los laboratorios a partir del repo privado.

Las fuentes con solución viven en `ml-unison/soluciones` (privado), con esta estructura:

    soluciones/
      unidades/01-introduccion/u01-lab01-....ipynb   (o .py en formato percent)

Cada fuente usa el formato de otter-grader (https://otter-grader.readthedocs.io):
celdas raw `# ASSIGNMENT CONFIG`, `# BEGIN QUESTION`, `# BEGIN SOLUTION`/`# END SOLUTION`
alrededor de las celdas de solución, `# BEGIN TESTS`/`# END TESTS`, y
y un bloque "BEGIN TEST CONFIG" con `hidden: true` para las pruebas ocultas.

Este script, para cada fuente:
  1. fuerza la configuración de Colab (pruebas en archivos descargados de este repo),
  2. ejecuta `otter assign`, que verifica que la solución pase todas las pruebas,
  3. copia la libreta del estudiante y sus pruebas públicas a unidades/<unidad>/laboratorios/,
  4. agrega la celda inicial del curso (RAMA, instalación de rdp).

Las respuestas y las pruebas ocultas nunca se copian a este repositorio.

Uso:
    uv run python herramientas/laboratorios.py --soluciones ../soluciones
    uv run python herramientas/laboratorios.py --soluciones ../soluciones u01-lab01
"""

from __future__ import annotations

import argparse
import json
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
OTTER = ["uvx", "--python", "3.12", "--with", "ipykernel", "--from", "otter-grader", "otter"]

CELDA_INICIAL = '''import os, sys

RAMA = "{rama}"  # rama del repositorio del curso de la que se toman paquete, datos y pruebas
os.environ["RDP_RAMA"] = RAMA
if "google.colab" in sys.modules:
    %pip install -q "rdp @ git+https://github.com/{repo}@{{RAMA}}#subdirectory=paquete"'''


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


def configurar(nb: nbformat.NotebookNode, url_pruebas: str) -> None:
    celda = next((c for c in nb.cells if c.cell_type == "raw" and c.source.lstrip().startswith("# ASSIGNMENT CONFIG")), None)
    if celda is None:
        raise ValueError("falta la celda raw '# ASSIGNMENT CONFIG'")
    cuerpo = celda.source.split("\n", 1)[1] if "\n" in celda.source else ""
    config = yaml.safe_load(cuerpo) or {}
    config.update({"runs_on": "colab", "export_cell": False, "generate": False, "solutions_pdf": False})
    config.setdefault("check_all_cell", True)
    config["tests"] = {**config.get("tests", {}), "files": True, "url_prefix": url_pruebas}
    celda.source = "# ASSIGNMENT CONFIG\n" + yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}


def procesar(fuente: Path, base_soluciones: Path, rama: str) -> Path:
    relativa = fuente.parent.relative_to(base_soluciones / "unidades")  # p. ej. 01-introduccion
    destino = RAIZ / "unidades" / relativa / "laboratorios"
    url = (f"https://raw.githubusercontent.com/{REPO}/{rama}/unidades/"
           f"{relativa.as_posix()}/laboratorios/tests/{fuente.stem}/")

    nb = leer(fuente)
    configurar(nb, url)
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        entrada = tmp / f"{fuente.stem}.ipynb"
        nbformat.write(nb, entrada)
        # otter assign ejecuta la solución y verifica que pase todas las pruebas
        r = subprocess.run([*OTTER, "assign", str(entrada), str(tmp / "dist"), "-v"], capture_output=True, text=True)
        estudiante = tmp / "dist" / "student" / entrada.name
        if r.returncode != 0 and not estudiante.exists():
            print(r.stdout[-3000:], r.stderr[-3000:], file=sys.stderr)
            raise RuntimeError(f"otter assign falló para {fuente.name}")

        alumno = nbformat.read(estudiante, as_version=4)
        nbformat.validator.normalize(alumno)
        texto = json.dumps(alumno)
        if "# BEGIN SOLUTION" in texto or "hidden: true" in texto:
            raise RuntimeError(f"{fuente.name}: la versión del estudiante conserva marcas de solución")
        inicial = nbformat.v4.new_code_cell(CELDA_INICIAL.format(rama=rama, repo=REPO))
        alumno.cells.insert(0, inicial)
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
