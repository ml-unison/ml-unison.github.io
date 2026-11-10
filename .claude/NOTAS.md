# Notas para continuar el trabajo con Claude Code

Última actualización: 2026-11-09. Este archivo está en un **repositorio público**: aquí no van
soluciones, enunciados de exámenes ni tokens. Lo privado vive en `ml-unison/soluciones`.

Al empezar una sesión en otra computadora, pídele a Claude: *«Lee `.claude/NOTAS.md` y
`presentaciones/NOTACION.md` antes de empezar»*. La memoria automática de Claude
(`~/.claude/projects/.../memory/`) **no viaja** entre computadoras; lo esencial está copiado aquí.
Si cambia algo importante, actualiza este archivo y haz commit.

Este archivo es el **estado actual y los acuerdos vigentes**. El historial de cada sesión (qué se hizo,
en qué computadora, qué quedó pendiente) va en `.claude/BITACORA.md`: al final de cada sesión se agrega
una entrada ahí y, si cambió el estado o un acuerdo, se actualiza también este archivo.

---

## 1. Preparar otra computadora

Los dos repositorios van **lado a lado** (las herramientas usan `../soluciones`):

```bash
mkdir -p ~/Documents/Cursos/ML && cd ~/Documents/Cursos/ML
git clone https://github.com/ml-unison/ml-unison.github.io.git
git clone https://github.com/ml-unison/soluciones.git          # privado: requiere `gh auth login`
cd ml-unison.github.io
uv sync --group dev && npm install
```

La ruta exacta no importa, solo que queden lado a lado. Computadoras conocidas: una Mac
(`~/Documents/Cursos/ML/`) y una Linux (`~/Documentos/Cursos/RdP-LCC/`). Si es una nueva, agrégala aquí.

Requisitos:
- [uv](https://docs.astral.sh/uv/), Node.js ≥ 24, `gh` autenticado con acceso a la organización `ml-unison`.
- macOS: `brew install libomp` (XGBoost lo necesita).
- Google Chrome instalado (capturas y revisión de láminas).
- Playwright para `herramientas/revisar_desborde.py`: **no está en `pyproject.toml`** y `uv sync`
  lo borra. Usar `uv run --with playwright python herramientas/revisar_desborde.py [u06]`.
- LaTeX (`latexmk`) para compilar exámenes en `../soluciones/examenes/`.

Comprobación rápida:

```bash
uv run pytest -q                       # marca, libretas, quizzes
node herramientas/construir.mjs        # todo a _construccion/
```

## 2. Estado actual (2026-11-04)

- El sitio rediseñado (Astro + Slidev) es el oficial desde el 2026-10-29: <https://ml-unison.github.io>.
  El profesor ya da clase con él (semestre **2026-2**).
- `main` y `rediseno-2027` están al mismo commit. Todo está publicado (U6 temas 3 y 4 desde el 2026-11-09).
- El material viejo (Jekyll) está en la etiqueta `legado-2026`.
- Las 8 unidades están producidas: 27 presentaciones, libretas de ejemplo, 8 laboratorios
  autocalificados (otter-grader) y 8 bancos de quiz.
- **Fase actual: revisión tema por tema**, según cómo va la clase y lo que necesitan los alumnos.
  Ya revisados a fondo:
  - U1 tema 1 (modelo de presentación aprobado).
  - Notación común en todas las presentaciones (`presentaciones/NOTACION.md`).
  - U5 tema 3, boosting (pseudo-residuos paso a paso, XGBoost, LightGBM y CatBoost).
  - U6 tema 1, rehecho: diferenciación automática y grafos de cómputo en lugar de retropropagación a
    mano; la red como ingeniería de características más un GLM en la última capa; Xavier/He con
    números. Lab 6 Parte A ahora es de `torch.autograd`.
  - U6 tema 3 dividido en dos (2026-11-06, publicado el 2026-11-09): `u06-03-convolucionales`
    (por qué imágenes, la parte convolucional como preprocesamiento, VGG, ResNet, Inception, Xception,
    EfficientNet) y `u06-04-transferencia-deteccion` (qué ve cada capa, última capa contra ajuste fino,
    detección de R-CNN a YOLO26). Secuencias pasó a `u06-05` (libreta `e07`). Libretas nuevas e04
    transferencia, e05 detección y e06 ajuste fino de YOLO con Global Wheat (solo Colab con GPU).
- **Examen intermedio 2026-2** (privado, `../soluciones/examenes/intermedio-2026-2/`): los 4 exámenes
  (A–D, dos problemas cada uno) están redactados, con material verificado y notas para calificar en su
  `README.md`. Falta confirmar con el profesor la versión final y cuándo se aplica en Teams.

## 3. Qué falta / siguientes pasos

1. U6 temas 3 y 4 ya publicados. Pendiente: correr en Colab con GPU
   `u06-e06-yolo-trigo` y subir su `resultados_yolo_trigo.json` si se quieren sus cifras en la lámina
   del trigo. Las fotos de `datos/fotos/` y `rdp.datos.archivo` ya están en `main`, así que Colab las encuentra.
2. Seguir revisando los temas que el profesor indique (la clase va por la U6). Cada revisión: libreta,
   módulo de datos de láminas, presentación, laboratorio y quiz si aplica, con cifras que coincidan.
3. Intermedio: revisión final con el profesor, compilar PDFs y aplicarlo en Teams.
4. Proyecto (competencias de Kaggle): entrega única **30/11/2026** y después citas para la defensa oral.
5. `refs/` (lista de cursos de DataCamp) está sin versionar; su contenido ya está en
   `sitio/src/datos/datacamp.ts`. Preguntar al profesor si se borra o se versiona.


## 4. Acuerdos de operación

- **Cada push a `main` publica de inmediato para los alumnos.** Cambios pequeños (una errata, una
  lámina): directo a `main`. Cambios grandes: en la rama `rediseno-2027`, que se publica en
  <https://ml-unison.github.io/preview/>, y merge a `main` **solo cuando el profesor lo apruebe**.
- Antes de hacer `git push` o merge, preguntar. Commits en español, descriptivos.
- Las libretas apuntan a `RAMA = "main"`. Para probar cambios en `rdp` o en `datos/` desde la vista
  previa: `uv run python herramientas/libretas.py rama rediseno-2027`, y de vuelta a `main` antes del merge.
- **Nada de soluciones en este repo** (`.gitignore` lo protege). Laboratorios con solución, exámenes,
  enunciados y autocalificadores: solo en `ml-unison/soluciones`.
- Ningún examen se publica en el sitio (rápidos, intermedio y parciales van en `soluciones/examenes/enunciados/`).
- Material 100 % propio: no enlazar presentaciones ni libretas de terceros; solo citar libros,
  artículos y documentación oficial.
- Si aparece una convención nueva que haya que homogeneizar en todo el curso, preguntar antes.

## 5. Decisiones tomadas (no reabrir sin que el profesor lo pida)

- **Quarto abandonado** (le parecía feo). Sitio en Astro (`sitio/`), presentaciones en Slidev
  (`presentaciones/`), que se abren en pestaña nueva. PyTorch para redes.
- Temario de 8 unidades: introducción, no supervisado, fundamentos, lineales, árboles y ensambles,
  redes neuronales, series de tiempo, MLOps.
- **Evaluación 2026-2:** actividades de aprendizaje (DataCamp + complementarias) 35 %, exámenes rápidos
  15 %, intermedio 20 %, proyecto 30 %. Rápidos e intermedio en Teams. Laboratorios y cuestionarios
  **no califican** (son práctica). Puntos extra por la ruta de DataCamp, sin cuantificar.
- Intermedio y proyecto con **defensa oral**: si no lo explica, se asume IA generativa y vale cero.
- **Proyecto:** competencia de Kaggle con repo GitHub abierto y leaderboard. Opciones: RSNA Knee,
  ROGII Wellbore, Soil Grain Size, Enveda CASMI 2026. Todas aceptan envíos tardíos. Descartadas: M5
  (vieja y con muchas soluciones publicadas) y Playground Series (demasiado fáciles).
- **Intermedio:** 4 exámenes × 2 problemas, individual, una semana, parámetros por expediente
  (`a` = último dígito, `b` = penúltimo, semilla = expediente), recuadros «Tú decides», bitácora de
  decisiones. Nivel matemático moderado: teoremas dados «sin demostrar» y solo demostraciones fáciles;
  integrales con tablas o SymPy. Datos descargados de la fuente original (UCI, etc.), no con `rdp.datos.cargar`.
- Datos de presas: fuente oficial datos.sonora.gob.mx («Recursos Hídricos», SAGARHPA, CC BY). El portal
  bloquea descargas automáticas: copias en `datos/` (ver `datos/README.md`); para actualizar, el
  profesor descarga a mano.

## 6. Estilo de las presentaciones (lo más importante)

El profesor rechazó la primera versión de la U1 por «espantosa» (diagramas de cajas, tecnicismos antes
de las ideas, sin ejemplos). Desde entonces:

- Pensadas para dar clase: empezar con preguntas al grupo y ejemplos concretos con datos reales;
  construir la intuición y **después** formalizar. Motivar cada concepto antes de definirlo.
- Nada de resultados técnicos «aventados» sin necesidad. Conforme avanza el curso, más formalismo y
  derivaciones (la U3 pidió más matemáticas: derivaciones y esbozos de prueba), sin perder la didáctica.
- Ilustraciones dibujadas (componentes Vue con datos reales), **nunca diagramas de cajas**.
  Layouts `pregunta`, `cita`, `caso`, `seccion`. Una idea por lámina. Notas para el profesor en
  comentarios HTML. Ética con casos reales y cifras citadas.
- **Autocontenidas:** no citan otras unidades ni otras presentaciones; sí libretas y laboratorios de la
  misma unidad. Todo símbolo que aparece en una lámina se explica en esa lámina.
- **Notación** (hoja completa en `presentaciones/NOTACION.md`): M datos, n atributos (índices desde 1),
  conjuntos 𝒟, instancia 𝐱^{(i)}, atributo x^{(i)}_j, parámetros θ, pesos w, sesgos b (nunca w₀),
  vectores y matrices en negritas, capas 𝐖^{[l]} con l normal, ℓ solo para la pérdida por ejemplo y
  L(θ) = (1/M)Σℓ, series y_t con t = 1..T.
- Cifras de las láminas = cifras de la libreta (las calcula el mismo código).

## 7. Cómo está hecho (convenciones técnicas)

- **Libretas:** se escriben en `.py` (jupytext *percent*) y se generan con
  `uv run python herramientas/libretas.py sincronizar [filtro]` (ids de celda estables). Encabezado del
  curso obligatorio (`tests/test_libretas.py`). Deben correr en CPU en pocos minutos. El CI solo
  ejecuta `unidades/*/ejemplos` (los laboratorios del alumno tienen funciones en blanco).
- **Datos de las láminas:** U1–U4 en `herramientas/datos_presentaciones.py` → `presentaciones/datos/uNN.json`;
  desde U5 un módulo por presentación `herramientas/laminas/uNN_MM.py` → `presentaciones/datos/uNN-MM.json`
  (`u06_01.py` ejecuta la libreta y toma sus cifras). Los JSON se versionan. Gráficas con Vue + d3-scale.
- **Paleta de gráficas:** azul, rojo, dorado #C9971C, gris (validada para daltonismo; no verde junto a
  rojo); etiquetas en color de texto. Colores definidos en `rdp/estilo.py`,
  `sitio/src/estilos/global.css` y `presentaciones/styles/tema.css` (los vigila `tests/test_marca.py`).
- **Identidad UNISON** (manual 2025–2030): azul #24398A, dorado #E1B73D; complementarios #20305A,
  #496EA5, #8DA3C9, #C0CEE3. Montserrat para títulos, Lato para cuerpo. El Depto. de Matemáticas no tiene
  logo: composición «DM · Departamento de Matemáticas» en Montserrat Bold.
  Manual: <https://identidadbuho.unison.mx/wp-content/uploads/2025/08/MANUAL-DE-IDENTIDAD_25-30_INICIAL-1.pdf>
- **Revisar láminas:** `node herramientas/revisar_katex.mjs [u06]` (fórmulas compilan) y
  `revisar_desborde.py [u06]` (nada se sale). Además, capturas con Chrome headless a 1280×720 con URL
  `#/N?clicks=10` y revisarlas una por una. Chrome headless tiene ancho mínimo de 500 px.
  `revisar_desborde.py` lee el total de láminas en `#/4`; si esa lámina no tiene pie, revisa 0 láminas.
- **Nueva presentación:** crear el `.md` con `browserExporter: true` y registrarla en
  `sitio/src/datos/materiales.ts`.
- **Quizzes:** `quizzes/banco/uNN.yml`, solo LaTeX estándar (sin macros de las láminas). Exportar a
  GIFT/Moodle XML/DOCX con `uv run python herramientas/exportar_quizzes.py`.
- **Laboratorios (otter-grader 7):** fuente en `../soluciones/unidades/<u>/uNN-labMM-*.py` con
  `# SOLUTION` o `# BEGIN/END SOLUTION` (y `""" # BEGIN PROMPT`); pruebas como `assert`; ocultas con
  `""" # BEGIN TEST CONFIG\nhidden: true\n""" # END TEST CONFIG`. Generar la versión del alumno:
  `uv run python herramientas/laboratorios.py --soluciones ../soluciones` (exige que la solución pase y
  que no se filtre nada; guarda el zip en `../soluciones/autocalificador/`). Calificar sin Docker: ver
  `../soluciones/README.md`.
- **Construcción:** `node herramientas/construir.mjs` (con `BASE=/preview/` para la vista previa).
  Flujos: `publicar.yml` (main → Pages), `preview.yml` (rediseno-2027 → `ml-unison/preview`, secreto
  `PREVIEW_TOKEN`), `verificar.yml` (pruebas y libretas).

## 8. Trampas conocidas

- Macro KaTeX `\T` ya incluye el `^`: escribir `X\T X`, nunca `X^\T` (sale texto crudo).
- En el frontmatter YAML, los valores con «: » van entre comillas.
- Las claves numéricas en JSON se reordenan: ordenar en el componente.
- `.katex` no hereda `text-transform` (h3 y th van en mayúsculas).
- sklearn ≥ 1.9: `LogisticRegression(penalty=None)` está obsoleto; usar `C=np.inf`.
- otter 7 solo descarga pruebas por URL en JupyterLite: `laboratorios.py` inserta una celda que las baja
  de GitHub a `tests/<lab>/`.
- Regenerar los JSON de redes neuronales tarda minutos (reentrenan modelos). `u06_04.py` tarda ~10 min
  y descarga ~750 MB (VGG-16, Imagenette, EuroSAT); incluye la detección (`_u06_04_deteccion.py`).
- Slidev registra los componentes al arrancar: si creas uno nuevo con el servidor corriendo, reinícialo.
- `ultralytics` (YOLO) es dependencia del proyecto; su licencia es AGPL-3.0.

## 9. Forma de trabajar que ha funcionado

- Para trabajo grande (una unidad o un tema completo): plan primero, aprobado por el profesor. El agente
  principal hace presentación y quiz; subagentes en paralelo hacen libretas y laboratorio con un
  *brief* común (estas convenciones + la retroalimentación del profesor) y reportan las cifras clave
  para que las láminas coincidan.
- Verificar antes de dar algo por terminado: pruebas, libretas ejecutadas, KaTeX, desborde y capturas.
- Al terminar, actualizar la sección 2 y 3 de este archivo y agregar la entrada en `BITACORA.md`.
