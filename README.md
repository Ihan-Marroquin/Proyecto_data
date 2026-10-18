# Proyecto 2: identificación de mosquitos

**Curso:** CC3084 — Data Science

**Universidad:** Universidad del Valle de Guatemala
**Integrantes:** Ihan Marroquin, Diego Fernando Patzán Marroquín y Milton Polanco

Proyecto reproducible sobre el conjunto **AI Mosquito Alert Challenge 2023**.
La primera etapa caracteriza la calidad, las clases y los *bounding boxes*. La
segunda etapa construye resultados parciales con tres modelos supervisados,
visualizaciones técnicas y una aplicación interactiva.

El sistema identifica seis etiquetas taxonómicas. Su salida es experimental y
no sustituye la validación de una persona experta en entomología.

## Archivos principales

- `notebooks/Proyecto_2_MosquitoAlert.ipynb`: análisis exploratorio ejecutado.
- `Proyecto2_informeFinal.pdf`: informe consolidado, con el análisis exploratorio como apéndice.
- `app.py`: aplicación Streamlit para clasificar, explorar y comparar modelos.
- `src/train_models.py`: entrenamiento, ajuste, evaluación y exportación de resultados.

## Modelos y evaluación

El preprocesamiento aplica el mismo flujo durante entrenamiento e inferencia:

1. Corrección de orientación EXIF.
2. Recorte del mosquito con la caja delimitadora y un margen de 5%.
3. Redimensionamiento a 96 × 96 mediante *letterbox*, sin deformar la imagen.
4. Extracción de HOG para forma e histogramas HSV para color.

Se comparan regresión logística, SVM lineal y bosque aleatorio. Cada familia
prueba dos configuraciones y se selecciona por F1 macro en validación. El modelo
elegido se reajusta con entrenamiento más validación y se informa una única
evaluación en prueba. Además de exactitud, se reportan:

- F1 macro y ponderado;
- exactitud balanceada;
- precisión, *recall* y F1 por clase;
- matrices de confusión normalizadas;
- tiempo de predicción y tamaño del modelo.

La porción de entrenamiento se limita por clase para reducir costo y moderar el
dominio de `albopictus` y `culex`. Validación y prueba conservan la distribución
estratificada de la muestra disponible. La corrida incluida en la entrega usa
2,728 imágenes completas del ZIP oficial; el EDA conserva la caracterización de
las 10,357 imágenes originales. Estos resultados son *baselines* parciales, no
una comparación definitiva con arquitecturas profundas.

### Resultados incluidos

| Modelo | F1 macro | Exactitud balanceada | Exactitud global |
|---|---:|---:|---:|
| Bosque aleatorio | 0.267 | 0.284 | 0.756 |
| Regresión logística | 0.207 | 0.203 | 0.524 |
| SVM lineal | 0.198 | 0.193 | 0.500 |

El bosque aleatorio es el mejor *baseline* de esta corrida por F1 macro. El
recall nulo en varias clases minoritarias impide interpretar la exactitud global
como una medida suficiente y motiva la siguiente iteración con más datos y
transferencia de aprendizaje.

## Aplicación

La aplicación incluye tres secciones:

- **Clasificar:** carga una imagen, permite seleccionar la región del mosquito y
  consulta un modelo o todos.
- **Explorar datos:** filtra clases y muestra distribuciones interactivas después
  de la limpieza.
- **Comparar modelos:** presenta métricas, *recall* por clase y errores. El
  usuario puede ocultar esta información.

Después de instalar dependencias y entrenar los modelos:

```text
streamlit run app.py
```

## Reproducción completa

### 1. Crear el entorno

```text
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

En macOS o Linux, la activación equivalente es `source .venv/bin/activate`.
Para reconstruir exactamente el entorno validado de la entrega en Python 3.12,
se puede usar `pip install -r requirements-lock.txt`.

### 2. Descargar el conjunto oficial

```text
python src/download_dataset.py
```

La descarga ocupa aproximadamente 10.4 GB. El ZIP se elimina después de
extraerlo y los datos no se versionan.

### 3. Entrenar y generar resultados

```text
python -m src.train_models
```

Para una prueba más corta:

```text
python -m src.train_models --max-train-per-class 250
```

El proceso guarda:

- modelos serializados en `models/`;
- tablas de evaluación en `models/evaluacion/`, utilizadas por la aplicación;
- cinco visualizaciones estáticas locales en `data/processed/figuras/`, sin versionar.

### 4. Ejecutar pruebas

```text
python -m unittest discover -s tests -v
```

## Estructura

```text
app.py
Proyecto2_informeFinal.pdf
notebooks/
├── Proyecto_2_MosquitoAlert.ipynb
data/
├── raw/                         # datos oficiales, no versionados
└── processed/                   # cachés de características, no versionadas
models/                          # artefactos para la aplicación
└── evaluacion/                  # métricas y predicciones para comparar modelos
src/
├── download_dataset.py
├── modeling.py
└── train_models.py
tests/
└── test_modeling.py
```

## Fuentes principales

- [Mosquito Alert en Zenodo](https://doi.org/10.5281/zenodo.15063886)
- [Dataset en Hugging Face](https://huggingface.co/datasets/mosquito-alert/ai-mosquito-alert-challenge-2023)

El conjunto se distribuye bajo licencia CC BY-NC-SA 4.0.
