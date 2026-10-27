// Material publicado de cada unidad. La página de la unidad se arma a partir de esto.
export interface Material {
  objetivos: string[]
  presentaciones: { id: string; titulo: string; descripcion: string; lista?: boolean }[]
  libretas: { ruta: string; titulo: string; descripcion: string }[]
  laboratorios: { ruta: string; titulo: string; descripcion: string }[]
  lecturas: { obra: string; detalle: string }[]
  quiz?: string
}

export const materiales: Record<string, Material> = {
  '01-introduccion': {
    objetivos: [
      'Explicar qué problemas se pueden resolver aprendiendo de datos y cuáles no.',
      'Formalizar un problema supervisado en términos de hipótesis, pérdida y riesgo.',
      'Distinguir el error en muestra ($E_{\\text{in}}$) del error fuera de muestra ($E_{\\text{out}}$).',
      'Construir un flujo de trabajo de scikit-learn sin fugas de información.',
      'Identificar riesgos éticos en los datos y en la elección de la métrica.',
    ],
    presentaciones: [
      { id: 'u01-01-aprender-de-datos', titulo: '¿Qué significa aprender de datos?', descripcion: 'Paradigmas, riesgo empírico, generalización, ciclo de vida y responsabilidad.', lista: true },
      { id: 'u01-02-pandas', titulo: 'Datos tabulares con pandas', descripcion: 'Selección, limpieza, agregación y uniones, pensando en el modelo que viene después.' },
      { id: 'u01-03-atributos', titulo: 'Ingeniería de atributos y pipelines', descripcion: 'Escalamiento, codificación, imputación y Pipeline de scikit-learn.' },
    ],
    libretas: [
      { ruta: 'unidades/01-introduccion/ejemplos/u01-e01-pandas-presas.ipynb', titulo: 'Explorar datos con pandas: las presas de Sonora', descripcion: 'Tipos de datos, uniones, agrupaciones, faltantes que no se ven y series de tiempo con datos de CONAGUA.' },
      { ruta: 'unidades/01-introduccion/ejemplos/u01-e02-preprocesamiento.ipynb', titulo: 'Preprocesamiento: preparar datos reales para un modelo', descripcion: 'Faltantes, categorías con y sin orden, logaritmos, fechas, escalas y atípicos, con las casas de Ames.' },
      { ruta: 'unidades/01-introduccion/ejemplos/u01-e03-pipeline-titanic.ipynb', titulo: 'Un primer pipeline sin trampas: el Titanic', descripcion: 'Qué información es legítima, línea base, Pipeline, una fuga de información real y la evaluación final.' },
    ],
    laboratorios: [
      { ruta: 'unidades/01-introduccion/laboratorios/u01-lab01-pipeline-censo.ipynb', titulo: 'Laboratorio 1 · Un pipeline completo, de los datos crudos a la evaluación', descripcion: 'Con el censo de EE. UU. de 1994: faltantes disfrazados, columnas redundantes, ColumnTransformer, línea base y evaluación final. Se verifica automáticamente.' },
    ],
    lecturas: [
      { obra: 'Abu-Mostafa, Magdon-Ismail y Lin (2012). Learning From Data.', detalle: 'Capítulo 1, The Learning Problem.' },
      { obra: 'James, Witten, Hastie, Tibshirani y Taylor (2023). An Introduction to Statistical Learning with Applications in Python.', detalle: 'Capítulo 2, Statistical Learning.' },
      { obra: 'Barocas, Hardt y Narayanan (2023). Fairness and Machine Learning.', detalle: 'Capítulo 1, Introduction.' },
    ],
    quiz: 'u01',
  },
}
