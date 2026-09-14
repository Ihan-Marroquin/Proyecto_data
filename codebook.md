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
