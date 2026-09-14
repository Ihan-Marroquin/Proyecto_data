# Proyecto 2: análisis exploratorio de Mosquito Alert

**Curso:** CC3084 — Data Science<br>
**Universidad:** Universidad del Valle de Guatemala<br>
**Integrantes:** Ihan Marroquin, Diego Fernando Patzán Marroquín y Milton Polanco

Análisis exploratorio del conjunto de datos **AI Mosquito Alert Challenge
2023**. El proyecto revisa la calidad de las anotaciones, describe la
distribución de las especies y estudia las dimensiones y posiciones de los
*bounding boxes*. No diagnostica enfermedades ni entrena un clasificador.

Todo el análisis reproducible se encuentra en:

```text
notebooks/Proyecto_2_MosquitoAlert.ipynb
```

## Contenido del cuaderno

1. Investigación y contexto del problema.
2. Descarga y carga reproducible del conjunto de datos.
3. Descripción de variables y tipos.
4. Limpieza y controles de calidad.
5. Preprocesamiento e ingeniería de variables.
6. Estadística descriptiva y distribución de clases.
7. Visualizaciones exploratorias.
8. Cruces de variables, posiciones y correlaciones.
9. Detección de valores atípicos.
10. Inspección visual de imágenes y conclusiones.

## Datos

El cuaderno descarga `mosquito_dataset_ai_v1.zip` desde la publicación oficial
de Mosquito Alert en Hugging Face. El archivo comprimido ocupa aproximadamente
10.4 GB y se elimina después de extraerlo. El conjunto analizado contiene
10,357 imágenes, sus anotaciones y seis clases.

Los datos no se versionan porque son grandes y se pueden reproducir desde la
fuente. Consulte [`codebook.md`](codebook.md) para la definición de las
variables, las clases y las observaciones de calidad.

## Cómo ejecutarlo

### Google Colab

Abra el cuaderno en Colab y ejecútelo de arriba hacia abajo. La primera
ejecución instala `huggingface_hub`, descarga el conjunto de datos en
`/content/mosquitoalert/` y guarda las salidas en
`/content/resultados_mosquitoalert/`.

### Entorno local

Desde la raíz del repositorio:

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
python src/download_dataset.py
jupyter notebook notebooks/Proyecto_2_MosquitoAlert.ipynb
```

El script de descarga guarda el conjunto en `data/raw/mosquitoalert/`. El
cuaderno detecta la ejecución local y escribe tablas y figuras en
`data/processed/`.

## Estructura del proyecto

```text
notebooks/
└── Proyecto_2_MosquitoAlert.ipynb   # análisis reproducible

data/
├── raw/                              # datos originales; no se versionan
└── processed/                        # tablas y figuras generadas

src/
└── download_dataset.py              # descarga y extrae el dataset

entregables/
├── Informe_Proyecto_2_MosquitoAlert.pdf
└── Presentacion_Proyecto_2_MosquitoAlert.pptx

codebook.md                              # variables, clases y fuente
requirements.txt                         # dependencias de Python
README.md                                # guía del proyecto
```

## Fuentes principales

- [Mosquito Alert / Zenodo](https://doi.org/10.5281/zenodo.15063886)
- [Dataset en Hugging Face](https://huggingface.co/datasets/mosquito-alert/ai-mosquito-alert-challenge-2023)
