"""Cuestionarios de autoevaluación dentro de libretas (Jupyter y Colab).

Usa el mismo banco de preguntas que el sitio: quizzes/banco/uNN.yml.
Los enunciados y opciones se muestran como Markdown (con LaTeX) y la
respuesta se elige con widgets por letra, que funcionan igual en Colab
y en Jupyter.

    import rdp
    rdp.quiz.mostrar("u01")                       # todo el banco
    rdp.quiz.mostrar("u01", ids=["u01-p02"])      # algunas preguntas
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import yaml

REPO = "ml-unison/ml-unison.github.io"
LETRAS = "abcdefghij"


def _rama() -> str:
    return os.environ.get("RDP_RAMA", "main")


def cargar_banco(nombre: str, rama: str | None = None) -> dict:
    """Lee el banco local (si la libreta está dentro del repo) o desde GitHub."""
    for base in (Path.cwd(), *Path.cwd().parents):
        p = base / "quizzes" / "banco" / f"{nombre}.yml"
        if p.exists():
            return yaml.safe_load(p.read_text(encoding="utf-8"))
    url = f"https://raw.githubusercontent.com/{REPO}/{rama or _rama()}/quizzes/banco/{nombre}.yml"
    with urllib.request.urlopen(url, timeout=30) as r:
        return yaml.safe_load(r.read().decode("utf-8"))


def correcta(p: dict, respuesta) -> bool:
    """Evalúa una respuesta (letra, conjunto de letras, bool o número)."""
    tipo = p.get("tipo", "opcion")
    if tipo == "opcion":
        ok = [LETRAS[i] for i, o in enumerate(p["opciones"]) if o.get("correcta")]
        return respuesta == ok[0]
    if tipo == "multiple":
        ok = {LETRAS[i] for i, o in enumerate(p["opciones"]) if o.get("correcta")}
        return set(respuesta or ()) == ok
    if tipo == "vf":
        return respuesta is p["respuesta"]
    if tipo == "numerica":
        try:
            v = float(str(respuesta).replace(",", "."))
        except (TypeError, ValueError):
            return False
        return abs(v - float(p["respuesta"])) <= float(p.get("tolerancia", 0)) + 1e-12
    raise ValueError(f"tipo desconocido: {tipo}")


def _widget_pregunta(n: int, p: dict):
    import ipywidgets as w
    from IPython.display import Markdown, display

    tipo = p.get("tipo", "opcion")
    texto = f"**{n}.** {p['enunciado'].strip()}"
    if tipo in {"opcion", "multiple"}:
        texto += "\n\n" + "\n".join(f"- **{LETRAS[i]})** {o['texto'].strip()}" for i, o in enumerate(p["opciones"]))
    salida_md = w.Output()
    with salida_md:
        display(Markdown(texto))

    letras = list(LETRAS[: len(p.get("opciones", []))])
    if tipo == "opcion":
        entrada = w.ToggleButtons(options=letras, value=None, style={"button_width": "3em"})
        leer = lambda: entrada.value
    elif tipo == "multiple":
        cajas = [w.Checkbox(description=l, indent=False, layout=w.Layout(width="4em")) for l in letras]
        entrada = w.HBox(cajas)
        leer = lambda: {c.description for c in cajas if c.value}
    elif tipo == "vf":
        entrada = w.ToggleButtons(options=[("Verdadero", True), ("Falso", False)], value=None)
        leer = lambda: entrada.value
    else:
        entrada = w.Text(placeholder="tu respuesta numérica", layout=w.Layout(width="14em"))
        leer = lambda: entrada.value

    boton = w.Button(description="Verificar", button_style="primary", layout=w.Layout(width="8em"))
    resultado = w.Output()

    def verificar(_):
        resultado.clear_output()
        r = leer()
        with resultado:
            if r in (None, "", set()):
                display(Markdown("_Elige o escribe una respuesta._"))
                return
            if correcta(p, r):
                msg = "✅ **Correcto.**"
                if p.get("explicacion"):
                    msg += " " + p["explicacion"].strip()
            else:
                msg = "❌ **Todavía no.**"
                if tipo == "opcion":
                    i = LETRAS.index(r)
                    if p["opciones"][i].get("retro"):
                        msg += " " + p["opciones"][i]["retro"].strip()
            display(Markdown(msg))

    boton.on_click(verificar)
    return w.VBox([salida_md, w.HBox([entrada, boton]), resultado],
                  layout=w.Layout(border="1px solid #C0CEE3", padding="6px 12px", margin="0 0 10px 0"))


def mostrar(nombre: str, ids: list[str] | None = None, rama: str | None = None) -> None:
    """Muestra el cuestionario `nombre` (p. ej. "u01"), opcionalmente filtrado por ids."""
    import ipywidgets as w
    from IPython.display import Markdown, display

    banco = cargar_banco(nombre, rama)
    preguntas = [p for p in banco["preguntas"] if ids is None or p["id"] in ids]
    display(Markdown(f"### 📝 {banco['titulo']}"))
    display(w.VBox([_widget_pregunta(n, p) for n, p in enumerate(preguntas, 1)]))
