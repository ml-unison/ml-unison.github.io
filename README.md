# Reconocimiento de Patrones · LCC · Universidad de Sonora

Sitio y material del curso de aprendizaje automático de la Licenciatura en Ciencias de la Computación (Departamento de Matemáticas, Universidad de Sonora).

- Sitio publicado: <https://ml-unison.github.io>
- Vista previa de la rama de desarrollo: <https://ml-unison.github.io/preview/>

## Estructura

| Ruta | Contenido |
|---|---|
| `_quarto.yml`, `_quarto-*.yml` | Configuración del sitio y perfiles `produccion` y `preview` |
| `_brand.yml` | Identidad visual (Manual de Identidad UNISON 2025–2030) |
| `_extensions/unison/` | Tema del sitio y de las presentaciones, macros LaTeX y los shortcodes `colab` y `quiz` |
| `unidades/NN-*/` | Página de cada unidad, con sus presentaciones (`slides/`), libretas (`ejemplos/`) y laboratorios |
| `quizzes/banco/uNN.yml` | Banco de preguntas: fuente única del sitio, las libretas y las exportaciones |
| `paquete/rdp/` | Paquete `rdp` que usan las libretas: estilo de gráficas, datos y cuestionarios |
| `herramientas/` | Scripts de mantenimiento |
| `datos/` | Conjuntos de datos que usan las libretas |
| `examenes/archivo/` | Exámenes de semestres anteriores |

## Trabajo local

Requisitos: [uv](https://docs.astral.sh/uv/) y [Quarto](https://quarto.org) ≥ 1.8.

```bash
uv sync --group dev                                 # entorno Python 3.12
export QUARTO_PYTHON=$PWD/.venv/bin/python
quarto preview                                      # sitio con recarga automática
quarto render --profile preview                     # igual que la vista previa publicada
uv run pytest -q                                    # pruebas de consistencia
uv run pytest --nbmake unidades/                    # ejecuta todas las libretas
```

### Libretas

Las libretas se escriben como `.py` en formato *percent*, que es fácil de revisar en git. Los `.ipynb` se generan sin salidas para abrirlos en Colab:

```bash
uv run python herramientas/libretas.py sincronizar  # .py → .ipynb
uv run python herramientas/libretas.py rama main    # antes del merge a main
```

Cada libreta define `RAMA = "..."`, la rama de la que Colab instala el paquete `rdp` y descarga los datos.

### Presentaciones

Son archivos `.qmd` con `format: unison-revealjs`. Tienen cajas académicas (`::: {.definicion nombre="..."}`, `.teorema`, `.algoritmo`, `.ejemplo`, `.idea`, `.cuidado`, `.pregunta`, `.demostracion`), figuras generadas con Python (`rdp.estilo.usar("presentacion")`) y widgets interactivos con OJS. Para obtener el PDF, abre la presentación con `?print-pdf` e imprime desde el navegador.

### Cuestionarios

Cada pregunta se escribe una sola vez en `quizzes/banco/uNN.yml`. Desde ahí:

- se muestra en el sitio con `{{< quiz uNN >}}`;
- se muestra en una libreta con `rdp.quiz.mostrar("uNN")`;
- se exporta a GIFT, Moodle XML y DOCX (para Microsoft Forms) con `uv run python herramientas/exportar_quizzes.py`.

En los bancos solo se permite LaTeX estándar, sin las macros del sitio.

### Laboratorios

Las versiones con solución viven en el repositorio privado `ml-unison/soluciones`, que usa [otter-grader](https://otter-grader.readthedocs.io). Aquí solo se publica la versión para estudiantes.

## Licencias

El código se distribuye bajo licencia MIT (ver `LICENSE`) y el material didáctico bajo [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.es). El escudo y el logotipo de la Universidad de Sonora son propiedad de la Universidad y se usan conforme a su Manual de Identidad Institucional. La tipografía Lato se distribuye bajo la licencia SIL OFL (`paquete/rdp/fuentes/OFL.txt`).

El material anterior al rediseño se conserva en la etiqueta `legado-2026`.
