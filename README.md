# Clasificación de Frutas mediante Redes Neuronales Convolucionales

## Introducción

Este proyecto implementa una Red Neuronal Convolucional (CNN) para la clasificación automática de frutas a partir de imágenes digitales. El objetivo principal es desarrollar un modelo de deep learning capaz de identificar con alta precisión 16 categorías distintas de frutas y vegetales, utilizando características visuales extraídas directamente de imágenes RGB preprocesadas.

## Objetivo
Desarrollar e implementar un modelo de Redes Neuronales Convolucionales para clasificación de imágenes, donde la variable objetivo ("y") corresponde a la etiqueta de la fruta o vegetal, y las variables de entrada ("x") son las matrices de píxeles de cada imagen.

# Descripción del dataset
El conjunto de datos base utilizado para este proyecto es el dataset ["Fruits-360"](https://www.kaggle.com/datasets/moltean/fruits/data) creado por Mihai Oltean, disponible públicamente en Kaggle. Se trata de una base de datos de alta calidad diseñada específicamente para la clasificación de alimentos.

Tiene un total de 260 categorías distintas que incluyen desde frutas comunes como manzanas, plátanos o peras, hasta frutos secos y vegetales exóticos como la cherimoya, pitahaya y kohlrabi.

# Estrategia de división
Esta división se fundamentó en una división inicial estándar de 80% para entrenamiento (Training) y 20% para prueba (Test). Sin embargo, para garantizar una correcta calibración de hiperparámetros, el bloque del 80% asignado a entrenamiento se subdividió, extrayendo un 10% del total del dataset exclusivamente para la validación (Validation). Esto resulta en una proporción efectiva de 70% Training, 10% Validation y 20% Test.

* 70% Training: Las redes neuronales requieren de una gran cantidad de datos para aprender características visuales. Al destinar la gran mayoría de los datos reales al entrenamiento, se maximiza la diversidad de imágenes y se facilita el aprendizaje de patrones morfológicos y cromáticos robustos.

* 10% Validation: Al asignar 10% para la validación, es lo suficientemente grande como para ser estadísticamente representativo de todas las frutas. Además, este porcentaje permite calcular de forma precisa el *accuracy*, permitiendo una correcta calibración de hiperparámetros.

* 20% Testing: La asignación responde a evaluar el modelo exclusivamente con datos reales. Esto asegura que las métricas finales no estén sesgadas por un tamaño de muestra insuficiente.

# Preprocesamiento de datos

## 1. Carga de datos
Se importó el conjunto de imágenes original directamente desde la plataforma Kaggle hacia el entorno de trabajo.

## 2. Estructuración y división de Conjuntos
Se reorganizaron los directorios aplicando la estrategia matemática de división: 80% original para training, subdividido en 70% training y 10% validation. Por último, 20% para el conjunto de test.

Como parte fundamental de la limpieza de datos, se realizó un proceso de agrupación para simplificar las etiquetas del modelo. El dataset original contenía clases demasiado específicas para una misma fruta. Por ejemplo, para la banana existían categorías independientes como Banana Yellow, Banana Red y Banana Lady Finger. Para resolver esto, se reestructuró el almacenamiento agrupando todos los elementos de una misma fruta en una sola carpeta contenedora, en este caso, Banana. Este proceso de categorización basado en el nombre base de la fruta se repitió con cada clase, permitiendo que el modelo aprenda a identificar la fruta de forma general.

## 3. Análisis Volumétrico Inicial

Se escaneó el dataset para contabilizar las imágenes disponibles para cada subconjunto. La cantidad inicial de imágenes antes del balanceo son:
- Training: 123,586 imágenes
- Validation: 13,635 imágenes
- Test: 45,724 imágenes

## 4. Diagnóstico Visual de Distribución

Se generaron gráficos de barras para ilustrar la cantidad exacta de imágenes por clase en cada subconjunto (Training, Validation y Test), marcando umbrales críticos de referencia.

Los umbrales fueron fundamentales porque actuaron como criterios de decisión: se eliminaron clases que no alcanzaban el límite inferior por falta de datos estadísticamente viables, y se recortaron aleatoriamente las clases que excedían el límite superior para garantizar una distribución equilibrada.

## 5.  Balanceo de datos
**Training:** El proceso de balanceo de datos se ejecutó mediante un límite inferior y un límite superior. Se eliminaron las clases que no alcanzaban el límite inferior por falta de datos. Y se recortó aleatoriamente las clases que excedían el límite superior.

**Validation y Test:** En estos subconjuntos se aplicó un límite máximo específico para recortar el excedente de imágenes en cada categoría. Posteriormente, se ejecutó un proceso de sincronización para alinear las clases de ambas particiones con las del conjunto de Training, eliminando categorías huérfanas.


### Umbrales de Referencia

**Training**: Para evitar sesgo de clase, los límites superior e inferior se establecieron en **3,000** y **2,000** imágenes respectivamente, garantizando una cantidad de muestras estadísticamente viable. Este rango representó el punto medio del dataset original, logrando una distribución equitativa de datos para un aprendizaje equilibrado del modelo. Como resultado, se eliminaron **79,335 imágenes** de clases sin suficientes datos.

**Validation**: Siguiendo una regla de tres proporcional con el subconjunto de entrenamiento, el umbral se calculó como: (3,000 × 10) / 70 ≈ **428 imágenes**. Como resultado, se eliminaron **12,771 imágenes** de las clases excedentes.

**Test**: Aplicando la misma metodología: (3,000 × 20) / 70 ≈ **857 imágenes**. Como resultado, se eliminaron **31,848 imágenes** de las clases excedentes.

Tras aplicar estos umbrales, se eliminó un total de **63 clases**, conservando solo las **16 clases** más robustas:
- Apple
- Avocado
- Blackberry
- Cherry
- Cucumber
- Grape
- Nut
- Onion
- Orange
- Peach
- Pear
- Pepper
- Plum
- Quince
- Raspberry
- Tomato

# Construcción del modelo

## Selección del Artículo del Estado de Arte

Alrashdi et al. (2026) utilizaron modelos de Deep Learning como CNN, DenseNet121, EfficientNetB3, Xception y ResNet50 para la detección y clasificación temprana de enfermedades en frutas [1]. Los autores demostraron que las CNN son capaces de extraer características visuales relevantes a partir de imágenes RGB preprocesadas, obteniendo altos niveles de precisión. Este trabajo sirve como base teórica para el presente proyecto, donde se emplea una CNN para la clasificación automática de frutas a partir de imágenes.

## Hiperparámetros Utilizados
- **Batch Size**: 64
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Funciones de Activación**: 
  - ReLU en capas convolucionales y densas
  - Softmax en la capa de salida

Es importante destacar que la selección de estos hiperparámetros se basó en las configuraciones reportadas en el trabajo de Alrashdi et al. [1], adaptadas para nuestro contexto específico de clasificación de frutas.

## Arquitectura del Modelo

La red se divide en dos bloques funcionales principales:

- **Feature Extraction**: Integrado por dos capas convolucionales 2D con kernels de 3×3, encargadas de la detección de patrones espaciales, bordes y texturas. Cada capa convolucional es seguida por MaxPooling 2×2 para reducir la dimensionalidad y proporcionar invariancia a la traslación.

- **Classification Stage**: Tras aplanar la información mediante una capa `Flatten`, los vectores de características pasan por tres capas densas secuenciales de 128, 64 y 32 neuronas (con activación ReLU), que reducen gradualmente la complejidad matemática. Finalmente, una capa de salida con 16 neuronas genera la distribución probabilística de las clases mediante Softmax.

## Métricas
Utilizamos las siguientes métricas para evaluar el desempeño del modelo de predicción:
- **Accuracy:** Mide la proporción de predicciones correctas que hace el modelo sobre imágenes que nunca vio durante el entrenamiento.
- **Loss function:** Es la métrica para saber que tan equivocado está el modelo con sus predicciones.
- **Precision:** Mide cuantas imágenes fueron clasificadas correctamente dentro de una categoría específica en relación con el total de elementos.
- **Recall:** Representa las imagenes que el modelo fue capaz de identificar y etiquetar correctamente en comparación con el total de frutas que existían verdaderamente.
- **F1-Score:** Es la media  balanceada entre `Precision` y `Recall`. Proporciona la evaluación más honesta y robusta sobre el rendimiento, garantizando que el modelo posee un equilibrio operativo sin sesgos predictivos.

# Resultados

## Matriz de Confusión

Para evaluar de manera exhaustiva el rendimiento del modelo, se generaron dos matrices de confusión: una global que agrupa todas las predicciones y otra multiclase de 16×16 que analiza el comportamiento ante cada categoría.

### Matriz de Confusión Global

Los datos obtenidos en el conjunto de prueba son:

| Métrica | Cantidad | Significado |
|----------|----------|----------|
| True Positives | 6,631 | Imágenes donde el modelo predijo correctamente la categoría de fruta. |
| True Negatives | 102,503 | Casos donde el modelo descartó correctamente las otras 15 clases no correspondientes. |
| False Positives | 217 | Instancias donde la red asignó erróneamente una fruta a una categoría incorrecta. |
| False Negatives | 217 | Casos donde imágenes de frutas reales fueron clasificadas en categorías incorrectas. |

**Análisis:** El desempeño del modelo es altamente eficiente. Un volumen de 6,631 verdaderos positivos frente a apenas 217 errores totales demuestra que la red neuronal posee una capacidad de generalización sobresaliente ante datos completamente nuevos. La precisión operativa asegura un riesgo mínimo de cometer clasificaciones erróneas en aplicaciones reales.

![Matriz de Confusión Global](general_confusion_matrix.png)

### Matriz de Confusión por Clases

La matriz multiclase de 16×16 permite observar el comportamiento específico del modelo ante cada categoría de fruta. La diagonal principal muestra altas tasas de verdaderos positivos por categoría, destacando clases como *Avocado* y *Blackberry* donde el modelo alcanzó una precisión del 100%.

**Punto de Mejora:** El principal desafío identificado es la distinción entre *Tomato* y *Pepper*, donde 84 tomates reales fueron clasificados erróneamente como pimientos. Los tomates evaluados presentaban morfologías ligeramente oblongas o alargadas; el extractor de bordes de la CNN generó descriptores de forma geométricamente similares a los de un pimiento regular, provocando asignación de probabilidades erróneas. Este es un área específica para futuras mejoras mediante data augmentation o arquitecturas más profundas.

![Matriz de Confusión por Clases](classes_confusion_matrix.png)

## Métricas de Desempeño

El modelo alcanzó los siguientes resultados en el conjunto de prueba (Test Set):

| Métrica | Valor | Interpretación |
|----------|----------|----------|
| **Loss (Categorical Crossentropy)** | 0.1468 | Bajo valor indica alta confianza en predicciones correctas |
| **Accuracy** | 0.9539 (95.39%) | El modelo acierta en 95 de cada 100 predicciones |
| **Precision** | 0.9539 (95.39%) | De las predicciones positivas, el 95.39% son correctas |
| **Recall** | 0.9539 (95.39%) | El modelo identifica correctamente el 95.39% de cada clase |
| **F1-Score** | 0.9539 (95.39%) | Balance perfecto entre precisión y recall |

**Análisis:** La alineación perfecta entre Accuracy, Precision, Recall y F1-Score demuestra que el clasificador alcanzó una simetría operativa ideal, sin sesgos hacia ninguna clase específica. El bajo valor de loss (0.1468) indica que el modelo asigna probabilidades muy altas y contundentes a las predicciones correctas. Esto demuestra que la red neuronal ha aprendido exitosamente a mapear los rasgos morfológicos y cromáticos con alto nivel de certeza matemática, sin comportarse como un clasificador que "adivina" aleatoriamente.

# Referencias

[1] Alrashdi, I., Sharawi, M., Ali, A.M. *et al.* Utilizing deep learning models for early detection and classification of fruit diseases: towards sustainable agriculture and enhanced food quality. *Sci Rep* **16**, 8167 (2026). https://doi.org/10.1038/s41598-026-38259-3