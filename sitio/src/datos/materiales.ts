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
  '03-fundamentos': {
    objetivos: [
      'Enunciar la desigualdad de Hoeffding, esbozar su demostración y explicar por qué no basta para una hipótesis elegida con los datos.',
      'Calcular funciones de crecimiento y dimensiones VC sencillas, e interpretar la cota VC.',
      'Derivar la descomposición en sesgo, varianza y ruido, y diagnosticar con curvas de aprendizaje.',
      'Derivar ridge, explicar por qué lasso selecciona variables y elegir la regularización con validación cruzada.',
      'Diseñar una validación honesta: validación cruzada, CV anidada y particiones por grupo.',
      'Elegir métricas y umbrales según los costos, evaluar la calibración y reconocer los límites de las métricas de equidad.',
    ],
    presentaciones: [
      { id: 'u03-01-factibilidad', titulo: '¿Es posible aprender?', descripcion: 'Hoeffding y su demostración, las 1,000 monedas, la cota de la unión, dicotomías, Sauer y la dimensión VC.' },
      { id: 'u03-02-sesgo-varianza', titulo: 'Sesgo, varianza y regularización', descripcion: 'La descomposición derivada paso a paso, curvas de aprendizaje, ridge, lasso y su lectura bayesiana.' },
      { id: 'u03-03-validacion', titulo: 'Validación', descripcion: 'El conjunto de validación, el optimismo de elegir, validación cruzada, LOO sin reentrenar, CV anidada y datos agrupados.' },
      { id: 'u03-04-metricas', titulo: 'Métricas', descripcion: 'Matriz de confusión, ROC y AUC, umbral de Bayes por costos, calibración y el resultado de imposibilidad en equidad.' },
    ],
    libretas: [
      { ruta: 'unidades/03-fundamentos/ejemplos/u03-e01-factibilidad.ipynb', titulo: '¿Es posible aprender? Hoeffding y dimensión VC, con simulaciones', descripcion: 'La urna, las 1,000 monedas, dicotomías contadas con programación lineal y la cota VC contra la brecha real.' },
      { ruta: 'unidades/03-fundamentos/ejemplos/u03-e02-sesgo-varianza.ipynb', titulo: 'Sesgo, varianza y regularización', descripcion: 'La descomposición simulada, curvas de aprendizaje, ridge y lasso con los datos de diabetes.' },
      { ruta: 'unidades/03-fundamentos/ejemplos/u03-e03-validacion.ipynb', titulo: 'Validación: estimar E_out sin hacer trampa', descripcion: 'Casas de Ames y facturas de una tienda: varianza de la validación, fórmula de LOO, regla de una desviación, CV anidada y GroupKFold.' },
      { ruta: 'unidades/03-fundamentos/ejemplos/u03-e04-metricas-credito.ipynb', titulo: 'Métricas de clasificación: el riesgo de crédito', descripcion: 'ROC, AUC por parejas, curva PR, umbral por costos, calibración y equidad por sexo con 30,000 clientes de Taiwán.' },
    ],
    laboratorios: [
      { ruta: 'unidades/03-fundamentos/laboratorios/u03-lab03-validacion-metricas.ipynb', titulo: 'Laboratorio 3 · Validación cruzada y métricas desde cero', descripcion: 'Ridge en forma cerrada, validación cruzada con la regla de una desviación, curva ROC, AUC por rangos y umbral por costos. Se verifica automáticamente.' },
    ],
    lecturas: [
      { obra: 'Abu-Mostafa, Magdon-Ismail y Lin (2012). Learning From Data.', detalle: 'Capítulos 1.3, 2 y 4.' },
      { obra: 'Hastie, Tibshirani y Friedman (2009). The Elements of Statistical Learning.', detalle: 'Secciones 3.4 y 7.10.' },
      { obra: 'Fawcett (2006). An introduction to ROC analysis. Pattern Recognition Letters 27.', detalle: 'Artículo completo.' },
    ],
    quiz: 'u03',
  },
  '04-lineales': {
    objetivos: [
      'Derivar las ecuaciones normales e interpretar mínimos cuadrados como proyección y como máxima verosimilitud.',
      'Elegir cómo calcular un ajuste lineal (QR, SVD, descenso por gradiente) según el condicionamiento del problema.',
      'Construir atributos (one-hot, interacciones) y diagnosticar un modelo lineal con sus residuos.',
      'Derivar la regresión logística, ajustarla con Newton (IRLS) e interpretar razones de momios.',
      'Reconocer la estructura común de los modelos lineales generalizados y usar la regresión de Poisson para conteos.',
      'Comparar clasificadores generativos (LDA, QDA, Bayes ingenuo) con discriminativos.',
      'Formular la SVM de margen máximo, su dual y el truco del kernel, y elegir $C$ y $\\gamma$ con validación cruzada.',
    ],
    presentaciones: [
      { id: 'u04-01-regresion-lineal', titulo: 'Regresión lineal', descripcion: 'Ecuaciones normales, proyección, Gauss–Markov, condicionamiento, descenso por gradiente, atributos y colinealidad.' },
      { id: 'u04-02-logistica-glm', titulo: 'Regresión logística y modelos lineales generalizados', descripcion: 'El Challenger, entropía cruzada, Newton como IRLS, separabilidad, softmax, familia exponencial y regresión de Poisson.' },
      { id: 'u04-03-generativos', titulo: 'Modelos generativos', descripcion: 'LDA, QDA y Fisher con los pingüinos; Bayes ingenuo para spam; generativo contra discriminativo.' },
      { id: 'u04-04-svm-kernels', titulo: 'Máquinas de vectores de soporte y kernels', descripcion: 'Margen máximo, dual y vectores de soporte, margen suave y pérdida bisagra, kernels polinomial y gaussiano.' },
    ],
    libretas: [
      { ruta: 'unidades/04-lineales/ejemplos/u04-e01-regresion-lineal.ipynb', titulo: 'Regresión lineal: cuántas bicicletas se rentarán', descripcion: 'Mínimos cuadrados desde cero, matriz sombrero, número de condición, descenso por gradiente, interacciones, colinealidad y residuos.' },
      { ruta: 'unidades/04-lineales/ejemplos/u04-e02-logistica-glm.ipynb', titulo: 'Regresión logística y GLM: del Challenger a las bicicletas', descripcion: 'Newton desde cero, razones de momios, bootstrap, separabilidad, softmax y regresión de Poisson.' },
      { ruta: 'unidades/04-lineales/ejemplos/u04-e03-generativos.ipynb', titulo: 'Modelos generativos: análisis discriminante y Bayes ingenuo', descripcion: 'LDA y QDA desde cero con los pingüinos, Fisher contra PCA, y un filtro de spam para SMS.' },
      { ruta: 'unidades/04-lineales/ejemplos/u04-e04-svm-kernels.ipynb', titulo: 'Máquinas de vectores de soporte y kernels', descripcion: 'El primal y el dual resueltos a mano, la cota de dejar-uno-fuera, margen suave, kernels y dígitos escritos a mano.' },
    ],
    laboratorios: [
      { ruta: 'unidades/04-lineales/laboratorios/u04-lab04-logistica-bayes.ipynb', titulo: 'Laboratorio 4 · Regresión logística y Bayes ingenuo desde cero', descripcion: 'Sigmoide estable, entropía cruzada, Newton con intersección y regularización, y Bayes ingenuo multinomial sobre matrices dispersas. Se verifica automáticamente.' },
    ],
    lecturas: [
      { obra: 'Abu-Mostafa, Magdon-Ismail y Lin (2012). Learning From Data.', detalle: 'Secciones 3.2 y 3.3, y el capítulo electrónico 8 (Support Vector Machines).' },
      { obra: 'Hastie, Tibshirani y Friedman (2009). The Elements of Statistical Learning.', detalle: 'Capítulos 3 y 4, y secciones 12.1–12.3.' },
      { obra: 'Ng y Jordan (2001). On discriminative vs. generative classifiers. NIPS 14.', detalle: 'Artículo completo.' },
    ],
    quiz: 'u04',
  },
}
