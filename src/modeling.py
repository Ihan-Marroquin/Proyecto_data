"""Preprocesamiento y extracción de características para Mosquito Alert.

El módulo comparte exactamente el mismo flujo entre entrenamiento y aplicación:
recorte del mosquito, corrección de orientación, redimensionamiento sin deformar,
HOG para forma y histogramas HSV para color.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from skimage.feature import hog


IMAGE_SIZE = 96
HISTOGRAM_BINS = 12
COLOR_FEATURE_LENGTH = HISTOGRAM_BINS * 3 + 6
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    annotations: Path
    images: Path


def find_dataset_paths(root: Path | str) -> DatasetPaths:
    """Encuentra ``annotations.csv`` e ``images`` aunque el ZIP cree una capa extra."""
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"No existe la ruta del dataset: {root}")

    annotation_files = sorted(root.rglob("annotations.csv"))
    if not annotation_files:
        raise FileNotFoundError(
            f"No se encontró annotations.csv dentro de {root}. "
            "Ejecute primero: python src/download_dataset.py"
        )

    annotations = annotation_files[0]
    likely = annotations.parent.parent / "images"
    if likely.is_dir():
        images = likely
    else:
        image_dirs = [
            path
            for path in root.rglob("images")
            if path.is_dir() and any(p.suffix.lower() in IMAGE_SUFFIXES for p in path.iterdir())
        ]
        if not image_dirs:
            raise FileNotFoundError(f"No se encontró una carpeta images válida dentro de {root}")
        images = image_dirs[0]
    return DatasetPaths(root=root, annotations=annotations, images=images)


def load_annotations(paths: DatasetPaths) -> pd.DataFrame:
    """Carga anotaciones, normaliza nombres y corrige cajas solo contra límites reales."""
    df = pd.read_csv(paths.annotations).copy()
    required = {
        "img_fName",
        "img_w",
        "img_h",
        "bbx_xtl",
        "bbx_ytl",
        "bbx_xbr",
        "bbx_ybr",
        "class_label",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Faltan columnas requeridas en annotations.csv: {missing}")

    df["class_label"] = df["class_label"].astype(str).str.strip().str.lower()
    df["image_path"] = df["img_fName"].map(lambda name: str(paths.images / str(name)))
    exists = df["image_path"].map(lambda value: Path(value).is_file())
    if not exists.all():
        examples = df.loc[~exists, "img_fName"].head(5).tolist()
        raise FileNotFoundError(
            f"Faltan {int((~exists).sum())} imágenes. Ejemplos: {examples}"
        )

    # Las tres anotaciones fuera de límites se corrigen al recortar. Se conserva
    # una bandera para que la decisión sea auditable.
    df["bbox_adjusted"] = (
        (df["bbx_xtl"] < 0)
        | (df["bbx_ytl"] < 0)
        | (df["bbx_xbr"] > df["img_w"])
        | (df["bbx_ybr"] > df["img_h"])
    )
    return df


def _safe_bbox(row: pd.Series, width: int, height: int, padding: float = 0.05) -> tuple[int, int, int, int]:
    x1 = float(row["bbx_xtl"])
    y1 = float(row["bbx_ytl"])
    x2 = float(row["bbx_xbr"])
    y2 = float(row["bbx_ybr"])

    if x2 <= x1 or y2 <= y1:
        return 0, 0, width, height

    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding
    x1 = max(0, min(width - 1, int(np.floor(x1 - pad_x))))
    y1 = max(0, min(height - 1, int(np.floor(y1 - pad_y))))
    x2 = max(x1 + 1, min(width, int(np.ceil(x2 + pad_x))))
    y2 = max(y1 + 1, min(height, int(np.ceil(y2 + pad_y))))
    return x1, y1, x2, y2


def crop_from_annotation(image: Image.Image, row: pd.Series, padding: float = 0.05) -> Image.Image:
    """Aplica EXIF, recorta una caja válida y devuelve RGB."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    return image.crop(_safe_bbox(row, image.width, image.height, padding=padding))


def crop_from_user_bbox(
    image: Image.Image,
    bbox: tuple[int, int, int, int] | None,
) -> Image.Image:
    """Recorta una entrada de la aplicación; sin caja usa la imagen completa."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    if bbox is None:
        return image
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(image.width - 1, int(x1)))
    y1 = max(0, min(image.height - 1, int(y1)))
    x2 = max(x1 + 1, min(image.width, int(x2)))
    y2 = max(y1 + 1, min(image.height, int(y2)))
    return image.crop((x1, y1, x2, y2))


def resize_letterbox(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """Redimensiona preservando proporción y rellena con el color mediano."""
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(image, dtype=np.uint8)
    fill = tuple(int(v) for v in np.median(array.reshape(-1, 3), axis=0))
    canvas = Image.new("RGB", (size, size), fill)
    left = (size - image.width) // 2
    top = (size - image.height) // 2
    canvas.paste(image, (left, top))
    return np.asarray(canvas, dtype=np.uint8)


def extract_feature_vector(image: Image.Image, size: int = IMAGE_SIZE) -> np.ndarray:
    """Concatena HOG y descriptores HSV; devuelve ``float32``."""
    rgb = resize_letterbox(image, size=size)
    gray = np.asarray(Image.fromarray(rgb).convert("L"), dtype=np.float32) / 255.0
    shape = hog(
        gray,
        orientations=9,
        pixels_per_cell=(12, 12),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)

    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float32) / 255.0
    color_parts: list[np.ndarray] = []
    for channel in range(3):
        hist, _ = np.histogram(hsv[..., channel], bins=HISTOGRAM_BINS, range=(0.0, 1.0))
        hist = hist.astype(np.float32)
        hist /= max(1.0, hist.sum())
        color_parts.append(hist)
    color_parts.append(hsv.mean(axis=(0, 1)).astype(np.float32))
    color_parts.append(hsv.std(axis=(0, 1)).astype(np.float32))
    color = np.concatenate(color_parts).astype(np.float32)
    return np.concatenate([shape, color]).astype(np.float32)


def select_feature_set(matrix: np.ndarray, feature_set: str) -> np.ndarray:
    if feature_set == "combined":
        return matrix
    if feature_set == "color":
        return matrix[:, -COLOR_FEATURE_LENGTH:]
    if feature_set == "hog":
        return matrix[:, :-COLOR_FEATURE_LENGTH]
    raise ValueError(f"Conjunto de características desconocido: {feature_set}")


def features_from_rows(
    rows: pd.DataFrame,
    *,
    image_size: int = IMAGE_SIZE,
    progress_every: int = 250,
) -> tuple[np.ndarray, list[str]]:
    """Extrae características de un conjunto de filas y reporta archivos dañados."""
    vectors: list[np.ndarray] = []
    valid_indices: list[str] = []
    failures: list[str] = []
    for count, (index, row) in enumerate(rows.iterrows(), start=1):
        try:
            with Image.open(row["image_path"]) as image:
                crop = crop_from_annotation(image, row)
                vectors.append(extract_feature_vector(crop, size=image_size))
                valid_indices.append(str(index))
        except Exception as exc:  # una imagen dañada no debe ocultar las demás
            failures.append(f"{row['img_fName']}: {exc}")
        if progress_every and count % progress_every == 0:
            print(f"  características: {count:,}/{len(rows):,}")
    if failures:
        preview = "\n".join(failures[:10])
        raise RuntimeError(f"Falló la lectura de {len(failures)} imágenes:\n{preview}")
    return np.vstack(vectors).astype(np.float32), valid_indices


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    scores -= scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_proba_compatible(model, features: np.ndarray) -> np.ndarray:
    """Obtiene probabilidades o una aproximación softmax para modelos lineales."""
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features), dtype=np.float64)
    if hasattr(model, "decision_function"):
        return softmax(model.decision_function(features))
    predictions = model.predict(features)
    classes = np.asarray(model.classes_)
    return np.eye(len(classes), dtype=np.float64)[np.searchsorted(classes, predictions)]


def iter_image_files(directory: Path | str) -> Iterable[Path]:
    directory = Path(directory)
    return (path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
