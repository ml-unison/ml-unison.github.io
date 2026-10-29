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
  '05-arboles-ensambles': {
    objetivos: [
      'Explicar y programar la construcción voraz de un árbol con Gini o entropía, y justificar por qué el error de clasificación es mala guía.',
      'Controlar la complejidad de un árbol con poda por costo–complejidad y validación cruzada, e identificar su alta varianza como motivación de los ensambles.',
      'Derivar y usar $\\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2$ para explicar por qué funcionan el bagging y los bosques, y elegir sus hiperparámetros con el error fuera de la bolsa.',
      'Derivar el voto de AdaBoost, los pseudo-residuos y el peso y la ganancia de una hoja en XGBoost, y programar gradient boosting desde cero.',
      'Entrenar y ajustar modelos de boosting (tasa de aprendizaje, número de árboles, parada temprana) y detectar fugas de información antes de comparar modelos.',
      'Calcular e interpretar la importancia por permutación, la dependencia parcial y las curvas ICE, y reconocer cómo las afectan los atributos correlacionados.',
      'Definir los valores de Shapley por sus axiomas, calcularlos por enumeración y explicar predicciones con SHAP, distinguiendo explicar el modelo de explicar el mundo.',
    ],
    presentaciones: [
      { id: 'u05-01-arboles', titulo: 'Árboles de decisión', descripcion: 'Cómo un árbol parte el espacio con preguntas de sí o no: impureza, el algoritmo voraz CART, árboles de regresión, poda por costo–complejidad y por qué los árboles son inestables.' },
      { id: 'u05-02-bosques', titulo: 'Bagging y bosques aleatorios', descripcion: 'Por qué promediar árboles reduce la varianza (y cuánto no), bootstrap, bagging, bosques aleatorios, error fuera de la bolsa e importancia de atributos.' },
      { id: 'u05-03-boosting', titulo: 'Boosting: de AdaBoost a XGBoost, LightGBM y CatBoost', descripcion: 'Pesos que se concentran en los errores, pseudo-residuos y descenso por gradiente en el espacio de funciones, parada temprana, y lo que hay dentro de XGBoost, LightGBM y CatBoost.' },
      { id: 'u05-04-interpretabilidad', titulo: 'Interpretabilidad: qué aprendió el modelo', descripcion: 'Por qué y cómo mirar dentro de un modelo: del asma que «protegía» a los valores de Shapley, con sus límites.' },
    ],
    libretas: [
      { ruta: 'unidades/05-arboles-ensambles/ejemplos/u05-e01-arboles.ipynb', titulo: 'Árboles de decisión: de los pingüinos a las bicicletas', descripcion: 'Un árbol de clasificación desde cero con Gini y árboles de regresión podados con validación cruzada que les ganan a los modelos lineales de la Unidad 4.' },
      { ruta: 'unidades/05-arboles-ensambles/ejemplos/u05-e02-bosques.ipynb', titulo: 'Bosques aleatorios: la sabiduría de muchos árboles', descripcion: 'Bagging y bosque desde cero en el censo, error fuera de la bolsa, sesgo y varianza medidos en las bicicletas, y la trampa de la importancia MDI.' },
      { ruta: 'unidades/05-arboles-ensambles/ejemplos/u05-e03-boosting.ipynb', titulo: 'Boosting y una campaña de mercadeo bancario', descripcion: 'AdaBoost y gradient boosting desde cero; una fuga de información a propósito; logística, bosque, HistGradientBoosting, XGBoost, LightGBM y CatBoost comparados; categorías codificadas sin fuga.' },
      { ruta: 'unidades/05-arboles-ensambles/ejemplos/u05-e04-interpretabilidad.ipynb', titulo: 'Interpretar un modelo de precios de casas', descripcion: 'Importancia por permutación, dependencia parcial, curvas ICE y valores de Shapley a mano, por enumeración exacta y con SHAP.' },
    ],
    laboratorios: [
      { ruta: 'unidades/05-arboles-ensambles/laboratorios/u05-lab05-arboles-boosting.ipynb', titulo: 'Laboratorio 5 · Árboles y boosting desde cero', descripcion: 'Impureza, la mejor división en O(N log N) y un árbol de regresión completo; bagging con error fuera de la bolsa, gradient boosting y las fórmulas de hoja y ganancia de XGBoost, en las bicicletas de Washington. Se verifica automáticamente.' },
    ],
    lecturas: [
      { obra: 'Hastie, Tibshirani y Friedman (2009). The Elements of Statistical Learning.', detalle: 'Secciones 9.2, 8.7, capítulo 10 y secciones 15.1–15.3.' },
      { obra: 'Breiman (2001). Random forests. Machine Learning 45.', detalle: 'Artículo completo.' },
      { obra: 'Chen y Guestrin (2016). XGBoost: A scalable tree boosting system. KDD.', detalle: 'Secciones 1–3.' },
      { obra: 'Molnar (2022). Interpretable Machine Learning, 2.ª ed.', detalle: 'Capítulos de dependencia parcial, importancia por permutación, ICE, valores de Shapley y SHAP.' },
    ],
    quiz: 'u05',
  },
  '08-mlops': {
    objetivos: [
      'Registrar experimentos de modo que cualquier corrida pueda compararse y reproducirse: código, datos, entorno, parámetros, métricas y artefactos.',
      'Administrar el ciclo de vida de un modelo con un registro (versiones, alias campeón/retador, firma) y documentarlo con una ficha que incluya métricas desagregadas por grupo.',
      'Construir y probar un servicio de predicción que valide entradas, reporte la versión del modelo y registre lo que predice.',
      'Distinguir los tipos de cambio de distribución, detectarlos con PSI y Kolmogorov–Smirnov, y explicar por qué el cambio de concepto exige medir el error con las etiquetas.',
    ],
    presentaciones: [
      { id: 'u08-01-experimentos', titulo: 'Experimentos reproducibles y registro de modelos', descripcion: 'Por qué el mismo código da números distintos, qué guardar de cada corrida, y cómo MLflow organiza experimentos, versiones, alias y firmas; más los riesgos de pickle y las fichas de modelo.' },
      { id: 'u08-02-despliegue-monitoreo', titulo: 'Despliegue y monitoreo', descripcion: 'Del caso Zillow a un servicio con FastAPI; cambio de covariables, de concepto y de prevalencia; PSI, la prueba de Kolmogorov–Smirnov y políticas de reentrenamiento.' },
    ],
    libretas: [
      { ruta: 'unidades/08-mlops/ejemplos/u08-e01-mlflow.ipynb', titulo: 'Seguimiento de experimentos con MLflow', descripcion: 'Siete modelos de riesgo de crédito registrados en MLflow, comparados con una consulta, promovidos con versiones y alias, reproducidos y documentados con su ficha.' },
      { ruta: 'unidades/08-mlops/ejemplos/u08-e02-despliegue-monitoreo.ipynb', titulo: 'Un servicio de predicción y la vigilancia del cambio', descripcion: 'Un modelo de bicicletas entrenado con 2011, servido con FastAPI y vigilado durante 2012 con PSI y Kolmogorov–Smirnov; políticas de reentrenamiento con y sin retraso de etiquetas.' },
    ],
    laboratorios: [
      { ruta: 'unidades/08-mlops/laboratorios/u08-lab08-servicio-monitoreo.ipynb', titulo: 'Laboratorio 8 · Un servicio de predicción vigilado', descripcion: 'PSI y Kolmogorov–Smirnov desde cero, un reporte de cambio por atributo, una API con FastAPI que valida su entrada y una política de reentrenamiento con ventana móvil. Se verifica automáticamente.' },
    ],
    lecturas: [
      { obra: 'Sculley et al. (2015). Hidden technical debt in machine learning systems. NeurIPS.', detalle: 'Artículo completo.' },
      { obra: 'Mitchell et al. (2019). Model cards for model reporting. FAT*.', detalle: 'Secciones 3 y 4.' },
      { obra: 'Huyen (2022). Designing Machine Learning Systems. O\'Reilly.', detalle: 'Capítulos 7 a 9.' },
      { obra: 'Documentación de MLflow y de FastAPI.', detalle: 'Tracking, Model Registry y el tutorial de FastAPI.' },
    ],
    quiz: 'u08',
  },
  '07-series-tiempo': {
    objetivos: [
      'Descomponer una serie y usar la autocorrelación para reconocer estacionalidad, memoria y no estacionariedad.',
      'Evaluar un pronóstico honestamente: contra referencias sencillas, por horizonte, con métricas adecuadas (MASE) y con validación de origen móvil, nunca barajando el tiempo.',
      'Identificar, ajustar y diagnosticar un modelo ARIMA o SARIMA con la ACF, la PACF, el AIC y la prueba de Ljung–Box, entendiendo un AR(p) como una regresión sobre los rezagos.',
      'Derivar la varianza del error de pronóstico de un AR(1) y de una caminata aleatoria, y evaluar intervalos de predicción.',
      'Construir atributos rezagados sin fuga, comparar estrategias multipaso por horizonte y obtener intervalos con la pérdida cuantil.',
    ],
    presentaciones: [
      { id: 'u07-01-series-validacion', titulo: 'Series de tiempo: estructura y validación', descripcion: 'Componentes y STL, autocorrelación, estacionariedad, pronósticos de referencia, MASE contra MAPE y validación hacia adelante, con la temperatura de Hermosillo y el caso de Google Flu Trends.' },
      { id: 'u07-02-arima', titulo: 'Modelos ARIMA', descripcion: 'Modelos con memoria: AR(1), un AR(p) como regresión sobre los rezagos, ACF y PACF, diferenciación, SARIMA por máxima verosimilitud y cómo crece la incertidumbre con el horizonte.' },
      { id: 'u07-03-aprendizaje-series', titulo: 'Pronosticar con aprendizaje automático', descripcion: 'Rezagos y calendario, la fuga de las variables exógenas, estrategias recursiva y directa, LightGBM y redes, e intervalos con pérdida cuantil.' },
    ],
    libretas: [
      { ruta: 'unidades/07-series-tiempo/ejemplos/u07-e01-series-validacion.ipynb', titulo: 'El clima de Hermosillo: tendencia, estacionalidad y cómo validar un pronóstico', descripcion: '33 años de temperatura diaria: ciclo anual, anomalías, autocorrelación, pronósticos de referencia por horizonte y por qué la validación cruzada aleatoria engaña con series.' },
      { ruta: 'unidades/07-series-tiempo/ejemplos/u07-e02-arima.ipynb', titulo: 'La curva de Keeling: pronosticar el CO₂ con ARIMA', descripcion: 'AR, MA y SARIMA desde sus propiedades con el CO₂ de Mauna Loa hasta 2014, elegidos por AIC, diagnosticados y comparados con lo que pasó hasta 2026.' },
      { ruta: 'unidades/07-series-tiempo/ejemplos/u07-e03-aprendizaje-series.ipynb', titulo: 'Pronosticar la demanda de bicicletas hora por hora', descripcion: 'Las próximas 24 horas con atributos rezagados sin fuga, origen móvil, LightGBM y una red en PyTorch, estrategias multipaso e intervalos con pérdida cuantil.' },
    ],
    laboratorios: [
      { ruta: 'unidades/07-series-tiempo/laboratorios/u07-lab07-pronosticos.ipynb', titulo: 'Laboratorio 7 · Validación temporal y modelos autorregresivos desde cero', descripcion: 'Origen móvil, pronósticos ingenuos, MASE, autocorrelación y un AR(p) con pronóstico recursivo para la temperatura de Hermosillo; cuánto engaña la validación cruzada barajada. Se verifica automáticamente.' },
    ],
    lecturas: [
      { obra: 'Hyndman y Athanasopoulos (2021). Forecasting: Principles and Practice, 3.ª ed. OTexts.', detalle: 'Capítulos 2, 3, 5, 9 y 10–12.' },
      { obra: 'Box, Jenkins, Reinsel y Ljung (2015). Time Series Analysis: Forecasting and Control, 5.ª ed.', detalle: 'Capítulos 3, 4 y 9.' },
      { obra: 'Makridakis, Spiliotis y Assimakopoulos (2022). M5 accuracy competition. International Journal of Forecasting 38.', detalle: 'Secciones 3 y 4.' },
    ],
    quiz: 'u07',
  },
  '06-redes-neuronales': {
    objetivos: [
      'Derivar e implementar la retropropagación en forma matricial para una red con softmax y entropía cruzada, y verificarla numéricamente.',
      'Explicar con argumentos de varianza por qué hacen falta la no linealidad y una buena inicialización (Xavier o He), y reconocer el desvanecimiento del gradiente.',
      'Explicar por qué el momento acelera en problemas mal condicionados y por qué Adam corrige el sesgo; elegir la tasa de aprendizaje con un barrido.',
      'Diagnosticar el sobreajuste con las curvas de validación y aplicar parada temprana, dropout y normalización por lotes.',
      'Calcular el tamaño de salida, los parámetros y el campo receptivo de una red convolucional, y explicar su ventaja sobre un perceptrón multicapa.',
      'Aplicar la transferencia con un modelo preentrenado, compararla con entrenar desde cero y reconocer los sesgos que se heredan.',
      'Medir un modelo de lenguaje con perplejidad, explicar el desvanecimiento del gradiente en las redes recurrentes y derivar la atención de producto punto escalado.',
    ],
    presentaciones: [
      { id: 'u06-01-mlp-backprop', titulo: 'Perceptrón multicapa y retropropagación', descripcion: 'Del perceptrón de 1958 a la retropropagación: qué hace una capa oculta, qué promete y qué no la aproximación universal, el gradiente en forma matricial y la inicialización.' },
      { id: 'u06-02-optimizacion', titulo: 'Entrenar redes: optimización y regularización', descripcion: 'Minilotes, tasa de aprendizaje, momento, Adam y calendarios; parada temprana, dropout, normalización por lotes y aumento de datos; el costo energético de entrenar.' },
      { id: 'u06-03-convolucionales', titulo: 'Redes convolucionales y transferencia', descripcion: 'Localidad y pesos compartidos: convolución, campo receptivo, pooling, conexiones residuales y transferencia, con Fashion-MNIST, EuroSAT y el caso Gender Shades.' },
      { id: 'u06-04-secuencias-atencion', titulo: 'Secuencias: redes recurrentes, atención y transformers', descripcion: 'Predecir la siguiente letra: perplejidad, n-gramas, el gradiente de las redes recurrentes, la LSTM, la atención y la razón del √d, y qué aprenden las cabezas de un transformer.' },
    ],
    libretas: [
      { ruta: 'unidades/06-redes-neuronales/ejemplos/u06-e01-mlp-backprop.ipynb', titulo: 'Una red neuronal desde cero, y luego con PyTorch', descripcion: 'Un perceptrón multicapa en NumPy con su retropropagación verificada con diferencias finitas, comparado con PyTorch en los dígitos; por qué importa la inicialización.' },
      { ruta: 'unidades/06-redes-neuronales/ejemplos/u06-e02-optimizacion.ipynb', titulo: 'Entrenar una red con Fashion-MNIST', descripcion: 'SGD, momento y Adam; el ruido del gradiente, el momento en un valle mal condicionado, parada temprana, dropout, normalización por lotes y más datos.' },
      { ruta: 'unidades/06-redes-neuronales/ejemplos/u06-e03-convolucionales.ipynb', titulo: 'Convoluciones con Fashion-MNIST y transferencia a imágenes de satélite', descripcion: 'La convolución programada y su equivariancia, una red convolucional contra un perceptrón multicapa, y una ResNet-18 de ImageNet aplicada a EuroSAT con pocos ejemplos.' },
      { ruta: 'unidades/06-redes-neuronales/ejemplos/u06-e04-secuencias-atencion.ipynb', titulo: 'Un modelo de lenguaje del Quijote, letra por letra', descripcion: 'De los n-gramas con suavizado de Laplace a una LSTM y un transformer mínimo escrito a mano, con perplejidad, temperatura y mapas de atención.' },
    ],
    laboratorios: [
      { ruta: 'unidades/06-redes-neuronales/laboratorios/u06-lab06-backprop.ipynb', titulo: 'Laboratorio 6 · Retropropagación desde cero y entrenamiento con PyTorch', descripcion: 'Capa lineal, ReLU y softmax con sus pasadas hacia atrás, verificación con diferencias finitas y un perceptrón multicapa con SGD; el ciclo de entrenamiento en PyTorch y una red convolucional pequeña. Se verifica automáticamente.' },
    ],
    lecturas: [
      { obra: 'Goodfellow, Bengio y Courville (2016). Deep Learning. MIT Press.', detalle: 'Capítulos 6–10.' },
      { obra: 'Prince (2023). Understanding Deep Learning. MIT Press.', detalle: 'Capítulos 6, 9, 10, 11 y 12.' },
      { obra: 'Vaswani et al. (2017). Attention is all you need. NeurIPS.', detalle: 'Secciones 3 y 4.' },
    ],
    quiz: 'u06',
  },
}
