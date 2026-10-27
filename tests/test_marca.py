"""Los colores institucionales deben ser los mismos en _brand.yml, SCSS y rdp.estilo."""
import re
from pathlib import Path

import yaml

from rdp.estilo import PALETA

RAIZ = Path(__file__).resolve().parents[1]
BRAND = yaml.safe_load((RAIZ / "_brand.yml").read_text(encoding="utf-8"))["color"]["palette"]


def test_paleta_python_coincide_con_brand():
    assert {k: v.upper() for k, v in PALETA.items()} == {k: v.upper() for k, v in BRAND.items()}


def test_scss_usa_colores_oficiales():
    oficiales = {v.upper() for v in BRAND.values()}
    for scss in (RAIZ / "_extensions/unison").rglob("*.scss"):
        for nombre, valor in re.findall(r"\$rdp-([\w-]+):\s*(#[0-9A-Fa-f]{6})", scss.read_text()):
            if nombre == "texto":
                continue
            assert valor.upper() in oficiales, f"{scss.name}: ${'rdp-' + nombre} = {valor} no está en _brand.yml"
