"""Los bancos de preguntas son válidos y usan solo LaTeX estándar."""
import re
import sys
from pathlib import Path

import pytest
import yaml

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "herramientas"))
import exportar_quizzes as ex  # noqa: E402
from rdp import quiz  # noqa: E402

BANCOS = sorted((RAIZ / "quizzes" / "banco").glob("u*.yml"))
# Macros que solo existen en el sitio (_extensions/unison/macros.html): no sirven en
# Colab ni en Moodle, así que están prohibidas en los bancos.
MACROS = re.findall(r"\\newcommand\{(\\\w+)\}", (RAIZ / "_extensions/unison/macros.html").read_text())


@pytest.mark.parametrize("ruta", BANCOS, ids=lambda p: p.stem)
def test_banco_valido(ruta):
    ex.cargar(ruta.stem)


@pytest.mark.parametrize("ruta", BANCOS, ids=lambda p: p.stem)
def test_sin_macros_del_sitio(ruta):
    texto = ruta.read_text(encoding="utf-8")
    usadas = [m for m in MACROS if re.search(re.escape(m) + r"(?![A-Za-z])", texto)]
    assert not usadas, f"{ruta.name} usa macros del sitio: {usadas}"


@pytest.mark.parametrize("ruta", BANCOS, ids=lambda p: p.stem)
def test_respuestas_correctas_se_aceptan(ruta):
    banco = yaml.safe_load(ruta.read_text(encoding="utf-8"))
    for p in banco["preguntas"]:
        tipo = p.get("tipo", "opcion")
        if tipo == "opcion":
            r = next(quiz.LETRAS[i] for i, o in enumerate(p["opciones"]) if o.get("correcta"))
        elif tipo == "multiple":
            r = {quiz.LETRAS[i] for i, o in enumerate(p["opciones"]) if o.get("correcta")}
        elif tipo == "vf":
            r = p["respuesta"]
        else:
            r = str(p["respuesta"])
        assert quiz.correcta(p, r), p["id"]


def test_exporta_todos_los_formatos(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "SALIDA", tmp_path)
    assert ex.main([]) == 0
    for ruta in BANCOS:
        for ext in ("gift", "xml", "docx"):
            assert (tmp_path / f"{ruta.stem}.{ext}").stat().st_size > 0
