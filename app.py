"""Aplicación Streamlit para explorar datos y comparar clasificadores."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

from src.modeling import (
    extract_feature_vector,
    find_dataset_paths,
    load_annotations,
    predict_proba_compatible,
    select_feature_set,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "entregables" / "resultados"
FIGURE_DIR = ROOT / "entregables" / "figuras"
DATASET_ROOT = ROOT / "data" / "raw" / "mosquitoalert"
CLASS_NAMES = {
    "aegypti": "Aedes aegypti",
    "albopictus": "Aedes albopictus",
    "anopheles": "Anopheles",
    "culex": "Culex",
    "culiseta": "Culiseta",
    "japonicus-koreicus": "Aedes japonicus / koreicus",
}
COLORS = ["#173A2B", "#2D5A45", "#7B9B83", "#C3913C", "#A64B38", "#66736C"]


st.set_page_config(
    page_title="Mosquito Alert",
    page_icon="🦟",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { --green:#173A2B; --gold:#C3913C; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: var(--green); }
    [data-testid="stMetricValue"] { color: var(--green); }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_models() -> dict[str, dict]:
    bundles = {}
    for path in sorted(MODEL_DIR.glob("*.joblib")):
        bundle = joblib.load(path)
        bundles[bundle["display_name"]] = bundle
    return bundles


@st.cache_data
def load_metrics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics = pd.read_csv(RESULT_DIR / "metricas_modelos.csv")
    per_class = pd.read_csv(RESULT_DIR / "metricas_por_clase.csv")
    predictions = pd.read_csv(RESULT_DIR / "predicciones_prueba.csv")
    return metrics, per_class, predictions


@st.cache_data
def load_exploration_data() -> pd.DataFrame | None:
    try:
        paths = find_dataset_paths(DATASET_ROOT)
        frame = load_annotations(paths)
    except (FileNotFoundError, ValueError):
        return None
    frame["bbox_width"] = frame["bbx_xbr"] - frame["bbx_xtl"]
    frame["bbox_height"] = frame["bbx_ybr"] - frame["bbx_ytl"]
    frame["bbox_ratio"] = (
        frame["bbox_width"].clip(lower=0)
        * frame["bbox_height"].clip(lower=0)
        / (frame["img_w"] * frame["img_h"])
    )
    frame["class_name"] = frame["class_label"].map(CLASS_NAMES)
    return frame


def model_ready() -> bool:
    return (RESULT_DIR / "metricas_modelos.csv").is_file() and any(MODEL_DIR.glob("*.joblib"))


def prediction_table(bundle: dict, image: Image.Image) -> pd.DataFrame:
    vector = extract_feature_vector(image, size=int(bundle["image_size"]))[None, :]
    vector = select_feature_set(vector, bundle["feature_set"])
    probabilities = predict_proba_compatible(bundle["model"], vector)[0]
    frame = pd.DataFrame({
        "clase": bundle["model"].classes_,
        "probabilidad": probabilities,
    })
    frame["nombre"] = frame["clase"].map(CLASS_NAMES).fillna(frame["clase"])
    return frame.sort_values("probabilidad", ascending=False)


def classifier_page(models: dict[str, dict]) -> None:
    st.header("Clasificación de una imagen")
    st.write(
        "Cargue una fotografía y, si es posible, encierre el mosquito con la caja. "
        "La aplicación aplica el mismo preprocesamiento usado durante el entrenamiento."
    )
    uploaded = st.file_uploader("Imagen JPG, JPEG o PNG", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        st.info("La predicción se habilita al cargar una imagen.")
        return

    original = ImageOps.exif_transpose(Image.open(uploaded)).convert("RGB")
    st.caption(f"Dimensiones originales: {original.width:,} × {original.height:,} píxeles")
    use_bbox = st.checkbox("Recortar el mosquito antes de clasificar", value=True)
    bbox = None
    if use_bbox:
        col1, col2 = st.columns(2)
        with col1:
            x1 = st.slider("X izquierda", 0, max(0, original.width - 1), 0)
            y1 = st.slider("Y superior", 0, max(0, original.height - 1), 0)
        with col2:
            x2 = st.slider("X derecha", 1, original.width, original.width)
            y2 = st.slider("Y inferior", 1, original.height, original.height)
        x2 = max(x1 + 1, x2)
        y2 = max(y1 + 1, y2)
        bbox = (x1, y1, x2, y2)

    preview = original.copy()
    if bbox:
        ImageDraw.Draw(preview).rectangle(bbox, outline="#C3913C", width=max(2, original.width // 250))
        crop = original.crop(bbox)
    else:
        crop = original
    left, right = st.columns([1.3, 1])
    with left:
        st.image(preview, caption="Entrada y región seleccionada", use_container_width=True)
    with right:
        st.image(crop, caption="Recorte que recibe el modelo", use_container_width=True)

    options = ["Todos los modelos", *models.keys()]
    choice = st.selectbox("Modelo", options)
    selected = models if choice == "Todos los modelos" else {choice: models[choice]}

    rows = []
    for name, bundle in selected.items():
        table = prediction_table(bundle, crop)
        winner = table.iloc[0]
        rows.append({
            "modelo": name,
            "predicción": winner["nombre"],
            "confianza": float(winner["probabilidad"]),
        })
        fig = px.bar(
            table.sort_values("probabilidad"),
            x="probabilidad",
            y="nombre",
            orientation="h",
            color="probabilidad",
            color_continuous_scale=[[0, "#E5E9E3"], [1, "#2D5A45"]],
            range_x=[0, 1],
            labels={"probabilidad": "Probabilidad", "nombre": "Grupo"},
            title=name,
        )
        fig.update_layout(coloraxis_showscale=False, height=360)
        st.plotly_chart(fig, use_container_width=True)

    summary = pd.DataFrame(rows)
    summary["confianza"] = summary["confianza"].map(lambda value: f"{value:.1%}")
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.warning(
        "Resultado experimental. Las clases minoritarias tienen pocos ejemplos y la predicción "
        "no sustituye la validación de una persona experta en entomología."
    )


def exploration_page() -> None:
    st.header("Exploración del conjunto limpio")
    frame = load_exploration_data()
    if frame is None:
        st.warning(
            "Los datos originales no están en este equipo. Ejecute "
            "`python src/download_dataset.py` para habilitar esta sección."
        )
        return

    a, b, c = st.columns(3)
    a.metric("Imágenes", f"{len(frame):,}")
    b.metric("Clases", frame["class_label"].nunique())
    c.metric("Cajas ajustadas", int(frame["bbox_adjusted"].sum()))

    selected_classes = st.multiselect(
        "Clases visibles",
        options=sorted(frame["class_name"].dropna().unique()),
        default=sorted(frame["class_name"].dropna().unique()),
    )
    filtered = frame[frame["class_name"].isin(selected_classes)]

    counts = filtered["class_name"].value_counts().rename_axis("clase").reset_index(name="imágenes")
    fig = px.bar(
        counts.sort_values("imágenes"),
        x="imágenes",
        y="clase",
        orientation="h",
        color="imágenes",
        color_continuous_scale=[[0, "#E5E9E3"], [1, "#173A2B"]],
        title="Distribución de clases",
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(
            filtered,
            x="class_name",
            y="bbox_ratio",
            color="class_name",
            color_discrete_sequence=COLORS,
            points=False,
            labels={"class_name": "Clase", "bbox_ratio": "Área de caja / área de imagen"},
            title="Escala relativa del mosquito",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        sample = filtered.sample(min(2_500, len(filtered)), random_state=42)
        fig = px.scatter(
            sample,
            x="img_w",
            y="img_h",
            color="class_name",
            color_discrete_sequence=COLORS,
            opacity=0.55,
            labels={"img_w": "Ancho", "img_h": "Alto", "class_name": "Clase"},
            title="Resolución de las fotografías",
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver variables y registros"):
        st.dataframe(
            filtered[[
                "img_fName", "class_label", "img_w", "img_h", "bbox_width", "bbox_height",
                "bbox_ratio", "bbox_adjusted",
            ]].head(1_000),
            hide_index=True,
            use_container_width=True,
        )


def performance_page() -> None:
    st.header("Rendimiento de los algoritmos")
    show_details = st.toggle("Mostrar métricas y visualizaciones técnicas", value=True)
    if not show_details:
        st.info("La información de rendimiento está oculta.")
        return
    metrics, per_class, predictions = load_metrics()

    best = metrics.sort_values("f1_macro", ascending=False).iloc[0]
    a, b, c = st.columns(3)
    a.metric("Mejor F1 macro", f"{best['f1_macro']:.3f}")
    b.metric("Exactitud balanceada", f"{best['exactitud_balanceada']:.3f}")
    c.metric("Modelo seleccionado", best["modelo"])

    tidy = metrics.melt(
        id_vars="modelo",
        value_vars=["f1_macro", "exactitud_balanceada", "exactitud"],
        var_name="métrica",
        value_name="valor",
    )
    tidy["métrica"] = tidy["métrica"].map({
        "f1_macro": "F1 macro",
        "exactitud_balanceada": "Exactitud balanceada",
        "exactitud": "Exactitud global",
    })
    fig = px.bar(
        tidy,
        x="valor",
        y="modelo",
        color="métrica",
        barmode="group",
        range_x=[0, 1],
        color_discrete_sequence=COLORS,
        labels={"valor": "Puntuación", "modelo": "", "métrica": "Métrica"},
        title="Comparación en la partición de prueba",
    )
    st.plotly_chart(fig, use_container_width=True)

    recall = per_class.pivot(index="clase", columns="modelo", values="recall")
    heat = go.Figure(data=go.Heatmap(
        z=recall.values,
        x=recall.columns,
        y=[CLASS_NAMES.get(value, value) for value in recall.index],
        zmin=0,
        zmax=1,
        colorscale="YlGn",
        text=np.round(recall.values, 2),
        texttemplate="%{text:.2f}",
        hovertemplate="Clase: %{y}<br>Modelo: %{x}<br>Recall: %{z:.3f}<extra></extra>",
    ))
    heat.update_layout(title="Recall por clase", height=460)
    st.plotly_chart(heat, use_container_width=True)

    selected_model = st.selectbox("Inspeccionar predicciones", metrics["modelo"].tolist())
    subset = predictions[predictions["modelo"] == selected_model]
    errors = subset[~subset["correcta"]].copy()
    st.write(
        f"{len(errors):,} errores de {len(subset):,} imágenes de prueba. "
        "La tabla permite identificar confusiones recurrentes."
    )
    st.dataframe(
        errors[["archivo", "real", "prediccion", "confianza"]].sort_values("confianza", ascending=False),
        hide_index=True,
        use_container_width=True,
    )


def main() -> None:
    st.title("Mosquito Alert")
    st.caption("Clasificación experimental y exploración del AI Mosquito Alert Challenge 2023")
    if not model_ready():
        st.error(
            "No se encontraron modelos o métricas. Ejecute `python -m src.train_models` "
            "desde la raíz del repositorio."
        )
        return
    models = load_models()
    page = st.sidebar.radio(
        "Sección",
        ["Clasificar", "Explorar datos", "Comparar modelos"],
    )
    st.sidebar.caption(
        "Proyecto académico de CC3084. Dataset con licencia CC BY-NC-SA 4.0."
    )
    if page == "Clasificar":
        classifier_page(models)
    elif page == "Explorar datos":
        exploration_page()
    else:
        performance_page()


if __name__ == "__main__":
    main()
