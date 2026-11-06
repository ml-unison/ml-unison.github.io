# Bitácora de trabajo con Claude Code

Registro cronológico de las sesiones: qué se hizo, qué se acordó y qué quedó pendiente. Complementa
a `.claude/NOTAS.md`, que es el **estado actual y los acuerdos vigentes** (si algo de aquí cambia un
acuerdo, se actualiza también allá). Repositorio público: aquí no van soluciones ni enunciados de
exámenes; de lo privado solo se anota *que* existe y dónde (`../soluciones`).

**Regla:** al final de cada sesión (o antes de cambiar de computadora) se agrega una entrada arriba de
las anteriores, con fecha, computadora, qué se hizo, acuerdos nuevos y pendientes, y se hace commit.

---

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
- Pendiente del punto 5 de NOTAS: aquí no existen los auxiliares de `examen-D` (no viajaron porque no
  estaban versionados), pero los de `examen-A/B/C` (`.aux`, `.log`, `.fls`, `.fdb_latexmk`, `.out`)
  **sí están versionados** en `soluciones`. Propuesta: agregar esos patrones al `.gitignore` de
  `soluciones` y quitarlos del índice (`git rm --cached`). Falta confirmarlo con el profesor.
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
