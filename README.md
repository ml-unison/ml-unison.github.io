# Reconocimiento de Patrones · LCC · Universidad de Sonora

Sitio y material del curso de aprendizaje automático de la Licenciatura en Ciencias de la Computación (Departamento de Matemáticas, Universidad de Sonora).

- Sitio publicado: <https://ml-unison.github.io>
- Vista previa de la rama de desarrollo: <https://ml-unison.github.io/preview/>

## Estructura

| Carpeta | Contenido |
|---|---|
| `sitio/` | Sitio del curso en [Astro](https://astro.build). Datos de las unidades en `src/datos/`, páginas en `src/pages/` |
| `presentaciones/` | Presentaciones en [Slidev](https://sli.dev): un `uNN-MM-nombre.md` por presentación, con el tema, las plantillas y las gráficas compartidas |
| `unidades/NN-nombre/` | Libretas de ejemplo (`ejemplos/`) y laboratorios para estudiantes (`laboratorios/`) |
| `quizzes/banco/` | Banco de preguntas, uno por unidad |
| `paquete/` | Paquete `rdp`: estilo de gráficas, carga de datos y quizzes en las libretas |
| `datos/` | Conjuntos de datos pequeños del curso |
| `examenes/archivo/` | Exámenes de semestres anteriores |
| `herramientas/` | Scripts de construcción, libretas, laboratorios, quizzes y datos de las presentaciones |

## Trabajo local

Requisitos: [uv](https://docs.astral.sh/uv/) y Node.js ≥ 24. En macOS, XGBoost necesita OpenMP: `brew install libomp`.

```bash
uv sync --group dev && npm install
npm run sitio                                         # sitio con recarga automática
npm run presentacion -- u01-01-aprender-de-datos.md   # una presentación con recarga automática
node herramientas/construir.mjs                       # todo junto en _construccion/
uv run pytest -q                                      # pruebas de consistencia
uv run pytest --nbmake -n 4 unidades/*/ejemplos       # ejecuta todas las libretas de ejemplo
node herramientas/revisar_katex.mjs [u04]             # todas las fórmulas de las presentaciones compilan
uv run python herramientas/revisar_desborde.py [u04]  # ninguna lámina se sale del área visible
```

`BASE=/preview/ node herramientas/construir.mjs` construye igual que la vista previa publicada. Las presentaciones siguen la notación de [`presentaciones/NOTACION.md`](presentaciones/NOTACION.md).

### Libretas

Las libretas se escriben como `.py` en formato *percent*, que es fácil de revisar en git. Los `.ipynb` se generan sin salidas para abrirlos en Colab:

```bash
uv run python herramientas/libretas.py sincronizar  # .py → .ipynb
uv run python herramientas/libretas.py rama main    # antes del merge a main
```

Cada libreta define `RAMA = "..."`, la rama de la que Colab instala el paquete `rdp` y descarga los datos.

### Presentaciones

Cada presentación es un archivo Markdown de Slidev sobre un lienzo de 1280 × 720. El tema (`presentaciones/styles/tema.css`) y las plantillas `portada` y `seccion` son comunes a todas. Para el contenido se usan:

- cajas académicas: `<Caja tipo="definicion" nombre="...">`, con los tipos `teorema`, `algoritmo`, `idea`, `cuidado`, `pregunta` y `correcto`;
- ecuaciones con KaTeX y las macros de `presentaciones/setup/katex.ts` (`\Ein`, `\Eout`, `\Hyp`, `\vx`…);
- gráficas como componentes Vue en `presentaciones/components/`. Los números los calcula Python y el navegador los dibuja con la tipografía de la lámina: las unidades 1–4 en `herramientas/datos_presentaciones.py` (→ `presentaciones/datos/uNN.json`) y, desde la Unidad 5, un módulo por presentación en `herramientas/laminas/uNN_MM.py` (→ `presentaciones/datos/uNN-MM.json`). Los JSON se versionan; se regeneran con `uv run python herramientas/datos_presentaciones.py [uNN_MM]` (los de redes neuronales tardan unos minutos porque reentrenan los modelos).

Para agregar una presentación a una unidad, crea el `.md` y regístralo en `sitio/src/datos/materiales.ts`. Se abre en una pestaña nueva; para el PDF, abre `…/#/export` en la presentación (o el menú «Exportar») e imprime desde el navegador. Cada presentación nueva debe tener `browserExporter: true` en su encabezado.

### Cuestionarios

Cada pregunta se escribe una sola vez en `quizzes/banco/uNN.yml`. Desde ahí:

- se muestra en la página de la unidad (campo `quiz` en `sitio/src/datos/materiales.ts`);
- se muestra en una libreta con `rdp.quiz.mostrar("uNN")`;
- se exporta a GIFT, Moodle XML y DOCX (para Microsoft Forms) con `uv run python herramientas/exportar_quizzes.py`.

En los bancos solo se permite LaTeX estándar, sin las macros de las presentaciones.

### Laboratorios

Las versiones con solución viven en el repositorio privado `ml-unison/soluciones`, que usa [otter-grader](https://otter-grader.readthedocs.io). Aquí solo se publica la versión para estudiantes.

### Publicación

Cada *push* a `main` construye y publica el sitio en https://ml-unison.github.io con el flujo `publicar.yml` (*Settings → Pages → Source* = «GitHub Actions»). Las libretas instalan `rdp` y descargan los datos de `main`.

Para probar un cambio grande antes de que lo vean los estudiantes, se trabaja en la rama `rediseno-2027`, que se publica en la vista previa (https://ml-unison.github.io/preview/) con cada *push*, y después se hace el merge a `main`. Las libretas de la vista previa siguen tomando `rdp` y los datos de `main`; para probar cambios en ellos, `uv run python herramientas/libretas.py rama rediseno-2027` en la rama (y de vuelta a `main` antes del merge).

El rediseño reemplazó por completo el sitio Jekyll anterior, que queda en la etiqueta `legado-2026`.

## Licencias

El código se distribuye bajo licencia MIT (ver `LICENSE`) y el material didáctico bajo [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.es). El escudo y el logotipo de la Universidad de Sonora son propiedad de la Universidad y se usan conforme a su Manual de Identidad Institucional. Las tipografías Montserrat, Source Sans 3, JetBrains Mono y Lato se distribuyen bajo la licencia SIL OFL.

El material anterior al rediseño se conserva en la etiqueta `legado-2026`.
