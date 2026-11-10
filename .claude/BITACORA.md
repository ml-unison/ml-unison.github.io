# Bitácora de trabajo con Claude Code

Registro cronológico de las sesiones: qué se hizo, qué se acordó y qué quedó pendiente. Complementa
a `.claude/NOTAS.md`, que es el **estado actual y los acuerdos vigentes** (si algo de aquí cambia un
acuerdo, se actualiza también allá). Repositorio público: aquí no van soluciones ni enunciados de
exámenes; de lo privado solo se anota *que* existe y dónde (`../soluciones`).

**Regla:** al final de cada sesión (o antes de cambiar de computadora) se agrega una entrada arriba de
las anteriores, con fecha, computadora, qué se hizo, acuerdos nuevos y pendientes, y se hace commit.

---

## 2026-11-09 · Linux (`~/Documentos/Cursos/RdP-LCC/`)

- Merge de `rediseno-2027` a `main` (avance directo, `4d94719`) a pedido del profesor: U6 temas 3 y 4
  (convolucionales; transferencia y detección con YOLO) y libretas e04–e07 publicadas. CI de la vista
  previa en verde antes del merge; «Publicar sitio» en `main` terminó con éxito.
- Pendiente: correr `u06-e06-yolo-trigo` en Colab con GPU para las cifras de la lámina del trigo.
- Fin de sesión: la siguiente será en una computadora nueva (ni la Mac ni esta Linux). Al llegar,
  seguir la sección 1 de `NOTAS.md` (clonar ambos repos lado a lado, `uv sync --group dev`,
  `npm install`, comprobación rápida) y anotar aquí la computadora y su ruta.

## 2026-11-06 · Linux (`~/Documentos/Cursos/RdP-LCC/`)

U6 tema 3 dividido en dos presentaciones, a pedido del profesor (plan aprobado). Todo en la rama
`rediseno-2027`, sin commit ni push todavía.

- **`u06-03-convolucionales`** (31 láminas): por qué importan las imágenes (2 billones de fotos al año,
  FDA, Sentinel-2, trigo), lo local importa también en audio, series y ADN; la red convolucional como
  preprocesamiento + clasificador (logística sobre atributos convolucionales congelados: 89.3 %, igual
  que la red completa; sobre pixeles, 83.6 %); VGG, ResNet (degradación, bloque, cuello de botella),
  Inception, Xception, EfficientNet y exactitud contra parámetros.
- **`u06-04-transferencia-deteccion`** (26 láminas, nueva): parches y máxima activación por capa de una
  VGG-16 (calculados por nosotros), dónde hay modelos preentrenados, tres formas de transferir con
  EuroSAT (k = 10/50/200: logística 77/86/89 %, última capa 78/87/89 %, ajuste fino 77/88/91 %, desde
  cero 55/63/66 %), Gender Shades, IoU, NMS, R-CNN → Faster R-CNN → YOLO, demo de YOLO26n con fotos CC
  de Hermosillo (Wikimedia Commons) y ajuste fino con Global Wheat.
- Secuencias renombrada a `u06-05` (libreta `u06-e07`). Libretas nuevas: e04 transferencia, e05
  detección, e06 ajuste fino de YOLO (Colab con GPU; en CI corre un modo rápido sintético).
- 16 componentes nuevos, 7 preguntas nuevas en el quiz u06 (p20–p26), `ultralytics` agregado al proyecto,
  `rdp.datos.archivo()` para archivos de `datos/` (fotos).
- Verificado: pytest, libretas de la U6 en modo CI, KaTeX, desbordes, construcción y capturas.
- Créditos de Colab: no hace falta comprarlos; la T4 gratuita alcanza para la libreta de YOLO (~1 h).
- Pendiente: revisión del profesor, commit, push a la vista previa y merge; correr e06 en Colab.

## 2026-11-05 · Linux (`~/Documentos/Cursos/RdP-LCC/`)

Primera sesión en esta computadora.

- Se leyeron `NOTAS.md` y `NOTACION.md` y se revisó todo el proyecto (sitio, 27 presentaciones,
  libretas, laboratorios, quizzes, herramientas, CI y el repo `soluciones`).
- Estado encontrado: `main` y `rediseno-2027` en el mismo commit (`9d10d70`); `soluciones` limpio
  (último commit `cc35b65`, Lab 6 con `torch.autograd`).
- Entorno de esta computadora: Node 26 (npm 11), `gh` autenticado y Chrome. Se instaló `uv` 0.13.7
  (instalador oficial, en `~/.local/bin`) y se corrieron `uv sync --group dev` y `npm install`:
  `uv run pytest -q` → 126 pasan, 8 omitidas; `node herramientas/construir.mjs` construye sin errores.
  `latexmk` instalado después por el profesor.
- Ojo: con npm 11, `npm install` agrega `engines` a `package-lock.json`. Es inocuo; se descarta con
  `git checkout package-lock.json` para no mezclarlo en los commits.
- Git no tenía identidad configurada; el profesor la configuró (`juliowaissman`,
  `julio.waissman@unison.mx`), igual que en sus commits anteriores.
- Limpieza de `soluciones` (aprobada): los auxiliares de LaTeX de `examen-A/B/C` estaban versionados;
  se agregaron los patrones de LaTeX al `.gitignore` y se sacaron del índice (siguen en disco).
  Commit `10bab66`, con push.
- Push de `main` con la bitácora (`ae10fca`); solo notas, no cambia nada para los alumnos.
- Acuerdo nuevo: llevar esta bitácora y actualizarla en cada sesión.

## Historia anterior (reconstruida de `git log` y de NOTAS)

- **2026-11-04** · U6 tema 1 rehecho: diferenciación automática y grafos de cómputo en lugar de
  retropropagación a mano; la red como ingeniería de atributos + GLM en la última capa; Xavier/He con
  números. Lab 6 Parte A con `torch.autograd`; tres preguntas nuevas en el quiz u06. Se escribieron
  las NOTAS para trabajar desde otra computadora. Intermedio 2026-2: exámenes A–D redactados (privado).
- **2026-10-29** · Sitio rediseñado publicado como oficial (merge a `main`). Notación común en todas las
  presentaciones (`NOTACION.md`); índice de capa `l` (ℓ solo para la pérdida). Boosting (U5 tema 3)
  revisado: pseudo-residuos paso a paso, bicicletas con RECM, XGBoost, LightGBM y CatBoost.
- **2026-10-27** · U2–U8 producidas, con sus datos. Evaluación 35/15/20/30 con DataCamp; proyecto en
  Kaggle (entrega 30/11/2026, defensa oral; M5 descartada). Se quitó la página de temario. Recursos en
  línea. CI con un hilo de OpenMP por libreta.
- **2026-10-26** · Rediseño: Quarto reemplazado por Astro + Slidev. U1 completa (presentación rehecha
  tras el rechazo de la primera versión), libretas con encabezado común, Lab 1 autocalificado. Presas
  desde el portal de datos abiertos de Sonora.
- **Antes de 2026-10** · Sitio Jekyll de semestres anteriores (etiqueta `legado-2026`).
