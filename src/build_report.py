"""Construye el informe final y agrega el EDA original como apéndice.

Requiere que ``python -m src.train_models`` haya generado métricas y figuras.
El PDF principal resume la metodología y los resultados; el informe exploratorio
anterior se conserva íntegro al final para mantener trazabilidad.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = ROOT / "entregables" / "resultados"
FIGURE_DIR = ROOT / "entregables" / "figuras"
SCREENSHOT_DIR = ROOT / "entregables" / "capturas"
LOGO = FIGURE_DIR / "logo_uvg.png"
SOURCE_EDA = ROOT / "entregables" / "Informe_Proyecto_2_MosquitoAlert.pdf"
OUTPUT = ROOT / "entregables" / "Informe_Final_Proyecto_2_MosquitoAlert.pdf"
BUILD_DIR = ROOT / "data" / "processed" / "report_build"
CORE_PDF = BUILD_DIR / "informe_resultados_core.pdf"

LIGHT = colors.HexColor("#B7B7B7")
CONTENT_WIDTH = A4[0] - 2 * inch


def format_decimal(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def load_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    required = [
        RESULT_DIR / "metricas_modelos.csv",
        RESULT_DIR / "metricas_por_clase.csv",
        RESULT_DIR / "ajuste_hiperparametros.csv",
        RESULT_DIR / "metadatos_ejecucion.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(
            "Faltan resultados. Ejecute primero `python -m src.train_models`.\n"
            + "\n".join(missing)
        )
    metrics = pd.read_csv(required[0])
    per_class = pd.read_csv(required[1])
    tuning = pd.read_csv(required[2])
    metadata = json.loads(required[3].read_text(encoding="utf-8"))
    return metrics, per_class, tuning, metadata


def styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=11,
            leading=14,
            alignment=TA_JUSTIFY,
            textColor=colors.black,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=9,
            leading=11,
            textColor=colors.black,
            alignment=TA_JUSTIFY,
        ),
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Times-Bold",
            fontSize=15,
            leading=19,
            alignment=TA_CENTER,
            textColor=colors.black,
            spaceAfter=0,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=base["BodyText"],
            fontName="Times-Italic",
            fontSize=12,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.black,
            spaceAfter=5,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Times-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.black,
            spaceBefore=14,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Times-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.black,
            spaceBefore=11,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["BodyText"],
            fontName="Times-Italic",
            fontSize=9,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.black,
            spaceBefore=4,
            spaceAfter=9,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=12,
            leading=18,
            alignment=TA_CENTER,
            textColor=colors.black,
        ),
        "cover_header": ParagraphStyle(
            "CoverHeader",
            parent=base["BodyText"],
            fontName="Times-Bold",
            fontSize=12,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.black,
            spaceAfter=5,
        ),
        "cover_date": ParagraphStyle(
            "CoverDate",
            parent=base["BodyText"],
            fontName="Times-Bold",
            fontSize=12,
            leading=15,
            alignment=TA_CENTER,
            textColor=colors.black,
        ),
        "report_title": ParagraphStyle(
            "ReportTitle",
            parent=base["BodyText"],
            fontName="Times-Bold",
            fontSize=12,
            leading=15,
            alignment=TA_LEFT,
            textColor=colors.black,
            spaceAfter=8,
        ),
        "table": ParagraphStyle(
            "Table",
            parent=base["BodyText"],
            fontName="Times-Roman",
            fontSize=8.5,
            leading=10.5,
            alignment=TA_LEFT,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Times-Bold",
            fontSize=8.5,
            leading=10.5,
            textColor=colors.black,
            alignment=TA_LEFT,
        ),
    }


def header_footer(canvas, doc) -> None:
    """La plantilla académica no utiliza encabezado, pie ni numeración visible."""


def build_table(data, widths, style_map, *, repeat_rows: int = 1) -> Table:
    total_width = sum(widths)
    if total_width > CONTENT_WIDTH:
        scale = CONTENT_WIDTH / total_width
        widths = [width * scale for width in widths]
    converted = []
    for row_index, row in enumerate(data):
        style_name = "table_header" if row_index == 0 else "table"
        converted.append([Paragraph(str(cell), style_map[style_name]) for cell in row])
    table = Table(converted, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.55, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    table.setStyle(TableStyle(commands))
    return table


def add_figure(story, path: Path, caption: str, style_map, width: float = 6.6 * inch) -> None:
    if not path.is_file():
        return
    image = Image(str(path))
    image._restrictSize(min(width, CONTENT_WIDTH), 5.5 * inch)
    story.append(KeepTogether([image, Paragraph(caption, style_map["caption"])]))


def build_core(metrics: pd.DataFrame, per_class: pd.DataFrame, tuning: pd.DataFrame, metadata: dict) -> None:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    s = styles()
    frame = Frame(inch, inch, CONTENT_WIDTH, A4[1] - 2 * inch, id="normal")
    doc = BaseDocTemplate(
        str(CORE_PDF),
        pagesize=A4,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
        title="Proyecto 2 Resultados de modelos para Mosquito Alert",
        author="Ihan Marroquin, Diego Patzan y Milton Polanco",
    )
    doc.addPageTemplates(PageTemplate(id="report", frames=frame, onPage=header_footer))
    story = []

    # Portada
    story.append(Paragraph("UNIVERSIDAD DEL VALLE DE GUATEMALA", s["cover_header"]))
    story.append(Paragraph("DATA SCIENCE - 40", s["subtitle"]))
    story.append(Paragraph("Erick Marroquin", s["cover_meta"]))
    story.append(Spacer(1, 0.28 * inch))
    if LOGO.is_file():
        logo = Image(str(LOGO), width=2.09 * inch, height=3.11 * inch)
        logo.hAlign = "CENTER"
        story.append(logo)
    story.append(Spacer(1, 0.48 * inch))
    story.append(Paragraph("Proyecto 2 Resultados parciales y visualizaciones técnicas", s["title"]))
    story.append(Spacer(1, 0.7 * inch))
    story.append(Paragraph("Ihan Marroquin  23108", s["cover_meta"]))
    story.append(Paragraph("Diego Patzan  23525", s["cover_meta"]))
    story.append(Paragraph("Milton Polanco  23471", s["cover_meta"]))
    story.append(Spacer(1, 0.72 * inch))
    story.append(Paragraph("GUATEMALA, 12 de octubre del 2026", s["cover_date"]))
    story.append(PageBreak())

    best = metrics.sort_values("f1_macro", ascending=False).iloc[0]
    story.append(Paragraph("Informe final", s["report_title"]))
    story.append(Paragraph("https://github.com/Ihan-Marroquin/Proyecto_data", s["body"]))
    story.append(Paragraph("Resumen", s["h1"]))
    story.append(Paragraph(
        "Este informe documenta la segunda etapa del proyecto de identificación visual de mosquitos. "
        "Se conservaron los hallazgos del análisis exploratorio y se implementó un flujo reproducible "
        "que recorta cada mosquito con su caja delimitadora, preserva la relación de aspecto, extrae "
        "descriptores HOG y HSV, ajusta tres familias de clasificadores y compara su rendimiento en una "
        "partición de prueba estratificada. Para esta corrida se usaron "
        f"<b>{metadata['registros_totales']:,}</b> imágenes de las "
        f"<b>{metadata.get('registros_dataset_original', metadata['registros_totales']):,}</b> documentadas "
        "en el análisis exploratorio. Los resultados se consideran parciales porque los modelos son líneas "
        "base de aprendizaje automático y no una arquitectura profunda afinada.",
        s["body"],
    ))
    story.append(Paragraph(
        f"El mejor resultado parcial correspondió a <b>{best['modelo']}</b>, con F1 macro de "
        f"<b>{format_decimal(best['f1_macro'])}</b> y exactitud balanceada de "
        f"<b>{format_decimal(best['exactitud_balanceada'])}</b>. La exactitud global se reporta, pero "
        "no dirige la selección debido a la relación aproximada de 98 a 1 entre la clase mayoritaria y la "
        "minoritaria.",
        s["body"],
    ))

    story.append(Paragraph("1 Introducción", s["h1"]))
    story.append(Paragraph(
        "Mosquito Alert reúne fotografías tomadas por ciudadanía en condiciones reales. Esa fuente amplía "
        "la cobertura de vigilancia, pero también introduce variaciones de fondo, iluminación, resolución, "
        "posición y escala. El problema de esta etapa consiste en determinar hasta qué punto modelos "
        "computacionalmente accesibles pueden separar seis etiquetas taxonómicas sin ocultar el desempeño "
        "de Aegypti y Anopheles, que cuentan con pocos ejemplos.",
        s["body"],
    ))
    story.append(Paragraph(
        "La aplicación construida permite cargar una fotografía, delimitar el mosquito, consultar cada modelo "
        "por separado o compararlos, explorar las variables limpias y mostrar u ocultar las métricas. La salida "
        "es experimental y requiere validación experta antes de cualquier uso en vigilancia entomológica.",
        s["body"],
    ))

    story.append(Paragraph("2 Objetivos", s["h1"]))
    story.append(Paragraph(
        "<b>Objetivo general.</b> Construir y comparar modelos reproducibles que clasifiquen imágenes del "
        "AI Mosquito Alert Challenge 2023 y presentar sus resultados en visualizaciones técnicas y una "
        "aplicación interactiva.",
        s["body"],
    ))
    objective_rows = [
        ["Objetivo específico", "Evidencia de cumplimiento"],
        ["Preparar entradas de imagen sin deformar la morfología.", "Recorte por caja, corrección EXIF y letterbox de 96 x 96."],
        ["Evaluar varios algoritmos con ajuste de parámetros.", "Tres familias y dos configuraciones por familia."],
        ["Comparar calidad, velocidad y errores por clase.", "F1 macro, exactitud balanceada, recall, matrices y latencia."],
        ["Permitir exploración y clasificación interactiva.", "Aplicación Streamlit con tres secciones."],
    ]
    story.append(build_table(objective_rows, [2.6 * inch, 4.2 * inch], s))

    story.append(Paragraph("3 Marco teórico y revisión bibliográfica", s["h1"]))
    story.append(Paragraph(
        "Las redes convolucionales aprenden jerarquías de textura y forma directamente de los píxeles. "
        "Motta et al. (2019) compararon LeNet, AlexNet y GoogLeNet sobre 4,056 imágenes de tres especies y "
        "obtuvieron su mejor resultado de prueba con GoogLeNet. Park et al. (2020) mostraron que las redes "
        "profundas también permiten relacionar activaciones con rasgos morfológicos en ocho especies.",
        s["body"],
    ))
    story.append(Paragraph(
        "En fotografías de ciencia ciudadana, Pataki et al. (2021) entrenaron ResNet50 con imágenes de "
        "Mosquito Alert y reportaron AUC ROC de 0.96 para reconocer Aedes albopictus. Adhane, Dehshibi y "
        "Masip (2021) aplicaron transferencia de aprendizaje y Grad-CAM a datos del mismo proyecto, con "
        "especial atención a bandas claras en patas, abdomen y tórax. Zhao et al. (2022) ampliaron la comparación "
        "a CNN y Transformers y mostraron el potencial de Swin Transformer en un conjunto balanceado y "
        "controlado. Estas investigaciones justifican una CNN transferida como siguiente fase, pero también "
        "muestran que el tipo de imagen, el equilibrio y el protocolo de validación impiden comparar porcentajes "
        "de forma directa entre estudios.",
        s["body"],
    ))
    story.append(Paragraph(
        "Para obtener resultados parciales en CPU, el proyecto usa HOG como representación de contornos y "
        "orientaciones locales, junto con histogramas HSV para color. La regresión logística y la SVM lineal "
        "establecen baselines discriminativos; el bosque aleatorio aporta una relación no lineal sobre los "
        "descriptores de color. Todos los modelos usan ponderación de clases cuando el algoritmo lo permite.",
        s["body"],
    ))

    story.append(Paragraph("4 Metodología", s["h1"]))
    story.append(Paragraph("4.1 Datos y preprocesamiento", s["h2"]))
    split_table = pd.read_csv(RESULT_DIR / "distribucion_particiones.csv")
    pivot = split_table.pivot(index="clase", columns="particion", values="imagenes").fillna(0).astype(int)
    split_rows = [["Clase", "Entrenamiento", "Validación", "Prueba"]]
    for class_name, row in pivot.iterrows():
        split_rows.append([
            class_name,
            f"{row.get('train', 0):,}",
            f"{row.get('validation', 0):,}",
            f"{row.get('test', 0):,}",
        ])
    story.append(build_table(split_rows, [2.3 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch], s))
    story.append(Spacer(1, 7))
    story.append(Paragraph(
        f"La división usa una semilla fija de {metadata['semilla']} y estratificación. La porción de entrenamiento "
        f"se limita a {metadata['maximo_entrenamiento_por_clase']} imágenes por clase para reducir costo y "
        "acercar el aporte de las clases mayoritarias, mientras validación y prueba conservan la distribución "
        "de la muestra de modelado. "
        f"Se ajustaron {metadata['cajas_ajustadas_a_limites']} cajas fuera de límites contra el ancho y alto "
        "reales de la imagen. El CSV completo se preservó por separado y la selección de entradas disponibles "
        "queda declarada en los metadatos de ejecución.",
        s["body"],
    ))
    story.append(Paragraph("4.2 Selección y ajuste", s["h2"]))
    tuning_rows = [["Modelo", "Configuración", "Características", "F1 macro validación"]]
    for row in tuning.sort_values(["modelo", "f1_macro_validacion"], ascending=[True, False]).itertuples():
        tuning_rows.append([
            row.modelo,
            row.configuracion,
            row.caracteristicas,
            format_decimal(row.f1_macro_validacion),
        ])
    story.append(build_table(tuning_rows, [1.55 * inch, 2.15 * inch, 1.35 * inch, 1.75 * inch], s))
    story.append(Spacer(1, 7))
    story.append(Paragraph(
        "Cada familia se seleccionó por F1 macro en validación. Luego se volvió a entrenar la configuración "
        "ganadora con entrenamiento más validación y se evaluó una sola vez en prueba. Este procedimiento "
        "reduce el riesgo de elegir parámetros usando el mismo conjunto con el que se informa el resultado.",
        s["body"],
    ))

    story.append(Paragraph("5 Resultados y análisis", s["h1"]))
    metric_rows = [[
        "Modelo", "F1 macro", "Exactitud balanceada", "Exactitud", "ms/imagen", "MB"
    ]]
    for row in metrics.sort_values("f1_macro", ascending=False).itertuples():
        metric_rows.append([
            row.modelo,
            format_decimal(row.f1_macro),
            format_decimal(row.exactitud_balanceada),
            format_decimal(row.exactitud),
            f"{row.ms_por_imagen:.2f}",
            f"{row.tamano_modelo_mb:.2f}",
        ])
    story.append(build_table(metric_rows, [1.65 * inch, 0.9 * inch, 1.35 * inch, 0.85 * inch, 1.0 * inch, 0.75 * inch], s))
    story.append(Spacer(1, 7))
    story.append(Paragraph(
        f"{best['modelo']} obtuvo la mayor F1 macro. La distancia entre exactitud global y exactitud balanceada "
        "muestra cuánto puede favorecer la primera a Albopictus y Culex. Por ello, la interpretación principal "
        "se realiza con promedios macro y recall por clase. El tiempo por imagen corresponde únicamente a la "
        "predicción de características ya extraídas; la lectura y el preprocesamiento agregan latencia en la aplicación.",
        s["body"],
    ))
    add_figure(story, FIGURE_DIR / "01_comparacion_metricas.png", "Figura 1. Comparación de las métricas principales en prueba.", s)
    add_figure(story, FIGURE_DIR / "02_recall_por_clase.png", "Figura 2. Recall por clase y por modelo.", s)
    add_figure(story, FIGURE_DIR / "03_matrices_confusion.png", "Figura 3. Matrices de confusión normalizadas por fila.", s)
    add_figure(story, FIGURE_DIR / "04_calidad_velocidad.png", "Figura 4. Relación entre calidad, velocidad y tamaño del artefacto.", s)

    best_per_class = per_class[per_class["modelo"] == best["modelo"]].sort_values("recall")
    weakest = best_per_class.iloc[0]
    strongest = best_per_class.iloc[-1]
    story.append(Paragraph("5.1 Errores y clases minoritarias", s["h2"]))
    story.append(Paragraph(
        f"En el modelo seleccionado, el recall más bajo correspondió a <b>{weakest['clase']}</b> "
        f"({format_decimal(weakest['recall'])}) y el más alto a <b>{strongest['clase']}</b> "
        f"({format_decimal(strongest['recall'])}). El soporte de prueba debe leerse junto con estos valores: "
        "Aegypti y Anopheles tienen pocos casos y, por tanto, unos cuantos aciertos o errores cambian la métrica "
        "de forma sustancial. Las matrices de confusión permiten ver si los errores se concentran en clases "
        "visualmente cercanas o simplemente se desplazan hacia las clases frecuentes.",
        s["body"],
    ))
    add_figure(
        story,
        FIGURE_DIR / "05_confianza_modelo_seleccionado.png",
        "Figura 5. Confianza aproximada en predicciones correctas e incorrectas.",
        s,
    )

    story.append(Paragraph("6 Aplicación", s["h1"]))
    story.append(Paragraph(
        "La aplicación Streamlit contiene tres secciones. Clasificar acepta una imagen, permite delimitar el "
        "mosquito y muestra las probabilidades o puntajes normalizados de uno o todos los modelos. Explorar datos "
        "presenta filtros y gráficas interactivas sobre clases, resolución y proporción de la caja. Comparar modelos "
        "muestra métricas, recall por clase y errores; un interruptor permite ocultar esa información. El código "
        "carga los mismos artefactos y funciones de preprocesamiento usados en entrenamiento.",
        s["body"],
    ))
    add_figure(
        story,
        SCREENSHOT_DIR / "app_clasificacion.png",
        "Figura 6. Vista de clasificación y comparación de modelos.",
        s,
        width=6.7 * inch,
    )
    add_figure(
        story,
        SCREENSHOT_DIR / "app_rendimiento.png",
        "Figura 7. Vista interactiva de rendimiento.",
        s,
        width=6.7 * inch,
    )

    story.append(Paragraph("7 Limitaciones", s["h1"]))
    story.append(Paragraph(
        "Los resultados no constituyen una estimación clínica ni una validación de campo. El conjunto combina "
        "etiquetas a nivel de especie, género y complejo de especies. La muestra parcial del archivo oficial, el "
        "número reducido de Aegypti y Anopheles, y el uso de descriptores manuales limitan la generalización. "
        "Además, el recorte de la caja usa una anotación conocida durante entrenamiento; una aplicación final "
        "debería incorporar detección automática o solicitar explícitamente la región de interés.",
        s["body"],
    ))
    story.append(Paragraph(
        "La siguiente iteración debe evaluar transferencia de aprendizaje con una red ligera, aumentación "
        "dirigida, focal loss o muestreo balanceado, calibración de probabilidades y validación repetida. También "
        "conviene comparar la clasificación del recorte contra la fotografía completa para medir la dependencia "
        "de la caja delimitadora.",
        s["body"],
    ))

    story.append(Paragraph("8 Conclusiones", s["h1"]))
    conclusions = [
        f"El proyecto implementó un flujo reproducible desde la anotación hasta la predicción y corrigió {metadata['cajas_ajustadas_a_limites']} cajas fuera de límites de la muestra únicamente durante el recorte.",
        f"La comparación seleccionó {best['modelo']} como mejor baseline por F1 macro, con {format_decimal(best['f1_macro'])} en prueba.",
        "Las métricas por clase mostraron diferencias que la exactitud global no refleja; por eso la evaluación debe mantener promedios macro y matrices normalizadas.",
        "Las visualizaciones técnicas relacionan rendimiento, errores, latencia y tamaño, en lugar de presentar una sola cifra de precisión.",
        "La aplicación permite explorar datos limpios, clasificar imágenes y comparar u ocultar resultados, pero conserva advertencias sobre el alcance experimental.",
        "Los baselines cumplen la etapa de resultados parciales y dejan una referencia verificable para entrenar una CNN transferida en la siguiente iteración.",
    ]
    for item in conclusions:
        story.append(Paragraph(f"• {item}", s["body"]))

    story.append(Paragraph("Referencias", s["h1"]))
    references = [
        "Adhane, G., Dehshibi, M. M., & Masip, D. (2021). A deep convolutional neural network for classification of Aedes albopictus mosquitoes. IEEE Access, 9, 72681-72690. https://doi.org/10.1109/ACCESS.2021.3079700",
        "Bartumeus, F., Garriga, J., Falk, M., & Mosquito Alert Expert Community. (2025). AI Mosquito Alert Challenge Dataset 2023 (Version v1) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.15063886",
        "Motta, D., Santos, A. A. B., Winkler, I., et al. (2019). Application of convolutional neural networks for classification of adult mosquitoes in the field. PLOS ONE, 14(1), e0210829. https://doi.org/10.1371/journal.pone.0210829",
        "Park, J., Kim, D. I., Choi, B., Kang, W., & Kwon, H. W. (2020). Classification and morphological analysis of vector mosquitoes using deep convolutional neural networks. Scientific Reports, 10, 1012. https://doi.org/10.1038/s41598-020-57875-1",
        "Pataki, B. A., Garriga, J., Eritja, R., Palmer, J. R. B., Bartumeus, F., & Csabai, I. (2021). Deep learning identification for citizen science surveillance of tiger mosquitoes. Scientific Reports, 11, 4718. https://doi.org/10.1038/s41598-021-83657-4",
        "Zhao, D., Wang, X., Zhao, T., et al. (2022). A Swin Transformer-based model for mosquito species identification. Scientific Reports, 12, 18664. https://doi.org/10.1038/s41598-022-21017-6",
    ]
    for reference in references:
        story.append(Paragraph(reference, s["small"]))
        story.append(Spacer(1, 4))

    doc.build(story)


def merge_eda() -> None:
    writer = PdfWriter()
    for page in PdfReader(str(CORE_PDF)).pages:
        writer.add_page(page)
    if SOURCE_EDA.is_file():
        for page in PdfReader(str(SOURCE_EDA)).pages:
            writer.add_page(page)
    with OUTPUT.open("wb") as stream:
        writer.write(stream)


def main() -> None:
    metrics, per_class, tuning, metadata = load_results()
    build_core(metrics, per_class, tuning, metadata)
    merge_eda()
    print(OUTPUT)


if __name__ == "__main__":
    main()
