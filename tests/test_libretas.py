"""Las libretas están sincronizadas con su fuente .py, sin salidas y apuntando a una rama coherente."""
import json
import re
from pathlib import Path

import jupytext
import pytest

RAIZ = Path(__file__).resolve().parents[1]
LIBRETAS = sorted(p for p in (RAIZ / "unidades").rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts)
RAMA_RE = re.compile(r'^RAMA = "([^"]+)"', re.M)


def fuentes(nb: dict) -> list[str]:
    return ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in nb["cells"]]


@pytest.mark.parametrize("ipynb", LIBRETAS, ids=lambda p: p.name)
def test_sin_salidas(ipynb):
    nb = json.loads(ipynb.read_text(encoding="utf-8"))
    con_salida = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and c.get("outputs")]
    assert not con_salida, f"celdas con salida: {con_salida} (usa herramientas/libretas.py limpiar)"


@pytest.mark.parametrize("ipynb", LIBRETAS, ids=lambda p: p.name)
def test_sincronizada_con_py(ipynb):
    py = ipynb.with_suffix(".py")
    if not py.exists():
        pytest.skip("sin fuente .py")
    a = [s.strip() for s in fuentes(json.loads(ipynb.read_text(encoding="utf-8")))]
    b = [s.strip() for s in fuentes(jupytext.reads(py.read_text(encoding="utf-8"), fmt="py:percent"))]
    assert a == b, "regenera con: uv run python herramientas/libretas.py sincronizar"


def test_todas_usan_la_misma_rama():
    ramas = {}
    for p in LIBRETAS:
        m = RAMA_RE.search("\n".join(fuentes(json.loads(p.read_text(encoding="utf-8")))))
        assert m, f"{p.name}: falta la línea RAMA = \"...\""
        ramas[p.name] = m.group(1)
    assert len(set(ramas.values())) <= 1, ramas
