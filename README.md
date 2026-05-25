# Nombre: Benjamín Arauz
# Materia: Desarrollo de aplicaciones avanzadas de ciencias computacionales (Gpo 301)

# Introducción
El presente proyecto se centra en la clasificación de imágenes utilizando. El objetivo principal es construir una red robusta capaz de identificar con alta precisión múltiples categorías.

## Objetivo
Desarrollar e implementar un modelo de Redes Neuronales Convolucionales para clasificación de imágenes, donde la variable objetivo ("y") corresponde a la etiqueta de la fruta o vegetal, y las variables de entrada ("x") son las matrices de píxeles de cada imagen.

# Descripción del dataset
El conjunto de datos base utilizado para este proyecto es el dataset ["Fruits-360"](https://www.kaggle.com/datasets/moltean/fruits/data) creado por Mihai Oltean, disponible públicamente en Kaggle. Se trata de una base de datos de alta calidad diseñada específicamente para la clasificación de alimentos.

Tiene un total de 260 categorías distintas que incluyen desde frutas comunes como manzanas, plátanos o peras, hasta frutos secos y vegetales exóticos como la cherimoya, pitahaya y kohlrabi.

# Estrategia de división
Esta división se fundamentó en una división inicial estándar de 80% para entrenamiento (Training) y 20% para prueba (Test). Sin embargo, para garantizar una correcta calibración de hiperparámetros, el bloque del 80% asignado a entrenamiento se subdividió, extrayendo un 10% del total del dataset exclusivamente para la validación (Validation). Esto resulta en una proporción efectiva de 70% Training, 10% Validation y 20% Test.

* 70% Training: Las redes neuronales requieren de una gran cantidad de datos para aprender características visuales. Al destinar la gran mayoría de los datos reales al entrenamiento, se maximiza la diversidad de imágenes que pueden.

* 10% Validation: Al asignar 10% para la validación, es lo suficientemente grande como para ser estadísticamente representativo de todas las frutas. Además, este porcentaje permite calcular de forma precisa el *accuracy*, permitiendo una correcta calibración de hiperparámetros.

* 20% Testing: La asignación responde a evaluar el modelo exclusivamente con datos reales. Esto asegura que las métricas finales no estén sesgadas por un tamaño de muestra insuficiente.

# Procedimiento
## Preprocesamiento de datos

### 1. Carga de datos
Se importó el conjunto de imágenes original directamente desde la plataforma Kaggle hacia el entorno de trabajo.

### 2. Estructuración y División de Conjuntos
Se reorganizaron los directorios aplicando la estrategia matemática de división: 80% original para training, subdividido en 70% training y 10% validation. Por último, 20% para el conjunto de test, unificando nombres y validando categorías.

### 3.  Análisis volumétrico inicial
Se escaneó el dataset para contabilizar las imágenes disponibles por cada categoría.

### 4.  Diagnóstico visual de distribución
Se generaron gráficos de barras para ilustrar la cantidad exacta de imágenes por clase en cada conjunto (Training, Validation y Test), marcando umbrales críticos de referencia.

#### Training
**Umbral de training**

El límite de 3000 imágenes para el conjunto de datos se estableció, porque existen clases como *Apple* que tenían muchas muestras y existían clases con muy pocos datos. 3000 imágenes es un número medio entre todas las cantidades de imágenes que se tienen. Si se dejaban intactas, la red neuronal desarrollaría un sesgo probabilístico.

#### Test
**Umbral de test**

Dado que el conjunto de test representa el 20% de la partición general, su umbral no se eligió arbitrariamente, sino que se derivó matemáticamente para mantener una relación con el conjunto de training:

Test Threshold = (3000 × 20) / 70 ≈ 857

#### Validation
**Umbral de validation**

Siguiendo la misma regla de proporción geométrica, el conjunto de validation adquiere un límite proporcional:

Validation Threshold = (3000 × 10) / 70 ≈ 428

### 5.  Balanceo y aumento de datos
Existieron algunas categorías que superaban el límite establecido, para ello se recortaron las clases mayoritarias que excedían el límite, y para training con *Data Augmentation* se generaron imágenes sintéticas para las clases minoritarias, alcanzando un balance perfecto de 3,000 imágenes por categoría.

> Nota: Se usan los límites calculados previamente para segmentar la cantidad de imágenes.

### 6.  Ajuste y sincronización
Se aplicaron recortes estrictos en los conjuntos de evaluación basándose en proporciones matemáticas. Finalmente, se sincronizaron las carpetas para eliminar clases huérfanas.
