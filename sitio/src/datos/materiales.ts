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
      { ruta: 'unidades/01-introduccion/ejemplos/u01-e01-pandas-presas.ipynb', titulo: 'Explorar datos con pandas: las presas de Sonora', descripcion: 'Tipos de datos, uniones, agrupaciones, faltantes que no se ven y series de tiempo con datos abiertos del Gobierno de Sonora.' },
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
  '02-no-supervisado': {
    objetivos: [
      'Formular el agrupamiento como un problema de optimización y explicar por qué k-medias converge a un mínimo local.',
      'Elegir entre k-medias, agrupamiento jerárquico, DBSCAN y mezclas gaussianas según la forma de los grupos y el tamaño de los datos.',
      'Derivar el algoritmo EM para mezclas gaussianas y relacionarlo con k-medias.',
      'Calcular PCA con la SVD e interpretar varianza explicada, componentes y reconstrucciones.',
      'Usar t-SNE y UMAP para explorar datos sin sobreinterpretar tamaños ni distancias.',
      'Evaluar un agrupamiento sin etiquetas y reconocer los riesgos de segmentar personas.',
    ],
    presentaciones: [
      { id: 'u02-01-agrupamiento', titulo: 'Agrupamiento', descripcion: 'k-medias paso a paso, mínimos locales, cómo elegir k, jerárquico con las presas de Sonora, DBSCAN y evaluación sin etiquetas.' },
      { id: 'u02-02-mezclas', titulo: 'Mezclas gaussianas y el algoritmo EM', descripcion: 'Modelos generativos, responsabilidades, EM y por qué funciona, tipos de covarianza y BIC.' },
      { id: 'u02-03-dimension', titulo: 'Reducción de dimensión', descripcion: 'La maldición de la dimensión, PCA con la SVD, t-SNE y UMAP, y cómo no leer sus mapas.' },
    ],
    libretas: [
      { ruta: 'unidades/02-no-supervisado/ejemplos/u02-e01-segmentacion-clientes.ipynb', titulo: 'Segmentar clientes con k-medias, jerárquico y DBSCAN', descripcion: 'De 25,900 facturas a una tabla RFM, k-medias programado a mano, elección de k, segmentos interpretables y atípicos.' },
      { ruta: 'unidades/02-no-supervisado/ejemplos/u02-e02-mezclas-pinguinos.ipynb', titulo: 'Mezclas gaussianas y el algoritmo EM', descripcion: 'Los pingüinos de Palmer: por qué falla k-medias, EM en 30 líneas de NumPy, responsabilidades, BIC y atípicos.' },
      { ruta: 'unidades/02-no-supervisado/ejemplos/u02-e03-reduccion-dimension.ipynb', titulo: 'Reducir dimensión: PCA, t-SNE y UMAP', descripcion: 'PCA desde cero con dígitos escritos a mano, componentes como imágenes, reconstrucciones y mapas no lineales.' },
    ],
    laboratorios: [
      { ruta: 'unidades/02-no-supervisado/laboratorios/u02-lab02-kmedias-pca.ipynb', titulo: 'Laboratorio 2 · k-medias y PCA desde cero', descripcion: 'Programar con NumPy el algoritmo de Lloyd y PCA con la SVD, verificarlos contra scikit-learn y cuantizar los colores de una foto. Se verifica automáticamente.' },
    ],
    lecturas: [
      { obra: 'James, Witten, Hastie, Tibshirani y Taylor (2023). An Introduction to Statistical Learning with Applications in Python.', detalle: 'Capítulo 12, Unsupervised Learning.' },
      { obra: 'Bishop (2006). Pattern Recognition and Machine Learning.', detalle: 'Capítulo 9, Mixture Models and EM; secciones 12.1 y 12.2, PCA.' },
      { obra: 'Wattenberg, Viégas y Johnson (2016). How to Use t-SNE Effectively. Distill.', detalle: 'Artículo interactivo completo.' },
    ],
    quiz: 'u02',
  },
}
