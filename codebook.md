# Codebook: AI Mosquito Alert Challenge 2023

El proyecto utiliza el conjunto de datos publicado por Mosquito Alert para una
tarea de reconocimiento visual de mosquitos. Cada registro de
`annotations.csv` corresponde a una imagen y contiene una etiqueta de clase y
un *bounding box*.

## Fuente

- Publicación: [Mosquito Alert / Zenodo](https://doi.org/10.5281/zenodo.15063886)
- Descarga reproducible: [Mosquito Alert en Hugging Face](https://huggingface.co/datasets/mosquito-alert/ai-mosquito-alert-challenge-2023)
- Archivo utilizado: `mosquito_dataset_ai_v1.zip`

Los datos se descargan con `src/download_dataset.py` o directamente desde el
cuaderno. No se almacenan en Git.

## Variables originales

| Variable | Tipo | Descripción |
|---|---|---|
| `img_fName` | texto | Nombre del archivo de imagen. |
| `img_w` | entero | Ancho de la imagen, en píxeles. |
| `img_h` | entero | Alto de la imagen, en píxeles. |
| `bbx_xtl` | entero | Coordenada X de la esquina superior izquierda del *bounding box*. |
| `bbx_ytl` | entero | Coordenada Y de la esquina superior izquierda del *bounding box*. |
| `bbx_xbr` | entero | Coordenada X de la esquina inferior derecha del *bounding box*. |
| `bbx_ybr` | entero | Coordenada Y de la esquina inferior derecha del *bounding box*. |
| `class_label` | texto | Clase taxonómica asignada al mosquito. |

Las coordenadas se expresan en píxeles respecto de la imagen original. El
origen se encuentra en la esquina superior izquierda.

## Variables derivadas

| Variable | Descripción |
|---|---|
| `image_area` | Área total de la imagen (`img_w × img_h`). |
| `image_aspect_ratio` | Relación entre ancho y alto de la imagen. |
| `bbox_width` | Ancho del *bounding box*. |
| `bbox_height` | Alto del *bounding box*. |
| `bbox_area` | Área del *bounding box*. |
| `bbox_ratio` | Proporción del área de la imagen ocupada por el *bounding box*. |
| `bbox_aspect_ratio` | Relación entre ancho y alto del *bounding box*. |
| `bbox_center_x_norm` | Posición horizontal normalizada del centro del *bounding box*. |
| `bbox_center_y_norm` | Posición vertical normalizada del centro del *bounding box*. |

## Clases y distribución observada

| Clase | Imágenes | Porcentaje |
|---|---:|---:|
| `albopictus` | 4,612 | 44.53 % |
| `culex` | 4,563 | 44.06 % |
| `culiseta` | 622 | 6.01 % |
| `japonicus-koreicus` | 429 | 4.14 % |
| `anopheles` | 84 | 0.81 % |
| `aegypti` | 47 | 0.45 % |

La distribución está fuertemente desbalanceada: la relación entre la clase
mayoritaria y la minoritaria es aproximadamente 98.13:1. Esta característica
debe considerarse antes de entrenar o evaluar modelos.

## Controles de calidad observados

- 10,357 registros y 10,357 imágenes.
- Sin valores faltantes ni filas duplicadas.
- Sin imágenes faltantes o carentes de anotación.
- Tres registros presentan coordenadas del *bounding box* fuera de las
  dimensiones declaradas de la imagen.
- Los valores atípicos se reportan para revisión visual; no se eliminan de
  manera automática.

## Variables usadas para modelado

Las variables tabulares no se pasan directamente a los clasificadores. Las
coordenadas del *bounding box* delimitan el mosquito durante la preparación de
la imagen; luego se extraen dos familias de descriptores:

| Descriptor | Dimensión | Uso |
|---|---:|---|
| HOG | 1,764 | Resume orientaciones locales, bordes y forma. |
| Histogramas HSV | 36 | Resume la distribución de tono, saturación y brillo. |
| Media y desviación HSV | 6 | Resume color e iluminación global. |

El vector combinado tiene 1,806 características. La regresión logística y la
SVM lineal usan el vector completo; el bosque aleatorio usa los 42 descriptores
de color para mantener un modelo compacto y diferente de los baselines lineales.

## Particiones y reproducibilidad

- Semilla fija: `42`.
- División predeterminada: 70% entrenamiento, 15% validación y 15% prueba.
- Estratificación por `class_label` en ambas divisiones.
- Máximo predeterminado: 600 imágenes de entrenamiento por clase.
- Validación y prueba mantienen la distribución de la muestra de modelado.
- La corrida incluida usa 2,728 imágenes completas de una descarga parcial del
  ZIP oficial. El EDA y `annotations_full.csv` conservan la referencia a los
  10,357 registros del conjunto original.
- Las cajas fuera de límites se recortan contra las dimensiones reales durante
  el preprocesamiento; las anotaciones completas se preservan por separado.

## Salidas de los modelos

| Archivo | Descripción |
|---|---|
| `metricas_modelos.csv` | Métricas globales, tiempo y tamaño del artefacto. |
| `metricas_por_clase.csv` | Precisión, *recall*, F1 y soporte por clase. |
| `ajuste_hiperparametros.csv` | Comparación de configuraciones en validación. |
| `predicciones_prueba.csv` | Predicción, confianza y probabilidades por imagen. |
| `metadatos_ejecucion.json` | Semilla, particiones, alcance y modelo seleccionado. |
