"""Administración de las libretas del curso.

Las libretas se escriben en formato *percent* (`.py`, más fácil de revisar en git)
y se generan como `.ipynb` sin salidas para Colab.

    uv run python herramientas/libretas.py sincronizar      # .py → .ipynb
    uv run python herramientas/libretas.py limpiar          # quita salidas de los .ipynb
    uv run python herramientas/libretas.py rama main        # fija RAMA = "main" (antes del merge)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import jupytext
import nbformat

RAIZ = Path(__file__).resolve().parents[1]
UNIDADES = RAIZ / "unidades"
RAMA_RE = re.compile(r'^(RAMA = ")[^"]+(")', re.M)


def _libretas() -> list[Path]:
    return sorted(p for p in UNIDADES.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)


def _fuentes_py() -> list[Path]:
    return sorted(p for p in UNIDADES.rglob("*.py") if p.with_suffix(".ipynb").exists() or "ejemplos" in p.parts or "laboratorios" in p.parts)


def limpiar(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    for c in nb.cells:
        if c.cell_type == "code":
            c.outputs = []
            c.execution_count = None
        c.metadata.pop("execution", None)
    nb.metadata.pop("widgets", None)
    nb.metadata["colab"] = {"provenance": [], "toc_visible": True}
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata.pop("jupytext", None)
    return nb


def cmd_sincronizar() -> None:
    for py in _fuentes_py():
        nb = limpiar(jupytext.read(py, fmt="py:percent"))
        destino = py.with_suffix(".ipynb")
        nbformat.write(nb, destino)
        print(f"✓ {destino.relative_to(RAIZ)}")


def cmd_limpiar() -> None:
    for p in _libretas():
        nbformat.write(limpiar(nbformat.read(p, as_version=4)), p)
        print(f"✓ {p.relative_to(RAIZ)}")


def cmd_rama(rama: str) -> None:
    for p in [*_fuentes_py(), *_libretas(), *sorted(UNIDADES.rglob("laboratorios/tests/**/*.py"))]:
        texto = p.read_text(encoding="utf-8")
        nuevo = RAMA_RE.sub(rf"\g<1>{rama}\g<2>", texto) if p.suffix == ".py" else \
            re.sub(r'(RAMA = \\")[^"\\]+(\\")', rf"\g<1>{rama}\g<2>", texto)
        # url de las pruebas de otter en los laboratorios
        nuevo = re.sub(r"(raw\.githubusercontent\.com/ml-unison/ml-unison\.github\.io/)[^/]+(/unidades/)",
                       rf"\g<1>{rama}\g<2>", nuevo)
        if nuevo != texto:
            p.write_text(nuevo, encoding="utf-8")
            print(f"✓ {p.relative_to(RAIZ)} → {rama}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sincronizar", help=".py (percent) → .ipynb sin salidas")
    sub.add_parser("limpiar", help="quita salidas de todos los .ipynb")
    r = sub.add_parser("rama", help="cambia la rama usada por las libretas")
    r.add_argument("rama")
    args = ap.parse_args(argv)
    if args.cmd == "sincronizar":
        cmd_sincronizar()
    elif args.cmd == "limpiar":
        cmd_limpiar()
    else:
        cmd_rama(args.rama)
    return 0


if __name__ == "__main__":
    sys.exit(main())
