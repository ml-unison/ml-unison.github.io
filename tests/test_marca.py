"""Los colores institucionales son los mismos en rdp.estilo (Python), el sitio y las presentaciones."""
import re
from pathlib import Path

import pytest

from rdp.estilo import PALETA

RAIZ = Path(__file__).resolve().parents[1]
OFICIALES = {v.upper() for v in PALETA.values()}
# Variables CSS que deben tomar su valor de la paleta oficial
INSTITUCIONALES = {"azul", "marino", "azul-medio", "azul-claro", "azul-tenue", "dorado", "rojo", "verde"}
HOJAS = [RAIZ / "sitio/src/estilos/global.css", RAIZ / "presentaciones/styles/tema.css"]


@pytest.mark.parametrize("hoja", HOJAS, ids=lambda p: p.parent.parent.name)
def test_css_usa_colores_oficiales(hoja):
    variables = dict(re.findall(r"--([\w-]+):\s*(#[0-9A-Fa-f]{6})", hoja.read_text(encoding="utf-8")))
    revisadas = INSTITUCIONALES & variables.keys()
    assert revisadas, f"{hoja.name} no define colores institucionales"
    for nombre in revisadas:
        assert variables[nombre].upper() in OFICIALES, f"{hoja.name}: --{nombre} = {variables[nombre]} no es un color oficial"


def test_azul_y_dorado_iguales_en_todos_lados():
    for hoja in HOJAS:
        texto = hoja.read_text(encoding="utf-8")
        assert f"--azul: {PALETA['azul']}" in texto
        assert f"--dorado: {PALETA['dorado']}" in texto
