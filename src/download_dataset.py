"""Descarga y extrae AI Mosquito Alert Challenge 2023.

El conjunto se guarda en ``data/raw/mosquitoalert/dataset``. El ZIP ocupa
aproximadamente 10.4 GB y se elimina al terminar la extracción.

Uso:
    python src/download_dataset.py
    python src/download_dataset.py --keep-zip
"""

import argparse
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "data" / "raw" / "mosquitoalert"
DATASET_DIR = BASE_DIR / "dataset"
IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

REPO_ID = "mosquito-alert/ai-mosquito-alert-challenge-2023"
FILENAME = "mosquito_dataset_ai_v1.zip"


def dataset_is_ready() -> bool:
    """Comprueba que existan las imágenes y al menos un CSV de etiquetas."""
    if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
        return False
    has_images = any(
        path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for path in IMAGES_DIR.iterdir()
    )
    return has_images and any(LABELS_DIR.glob("*.csv"))


def download_dataset(keep_zip: bool = False) -> Path:
    """Descarga el ZIP, lo extrae y devuelve la ruta del dataset."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Falta huggingface_hub. Instale primero: "
            "pip install -r requirements.txt"
        ) from exc

    if dataset_is_ready():
        print(f"Dataset ya disponible en {DATASET_DIR}")
        return DATASET_DIR

    BASE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Descargando {FILENAME} desde {REPO_ID}...")
    downloaded = Path(
        hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="dataset",
            local_dir=BASE_DIR,
        )
    )

    print(f"Extrayendo el dataset en {DATASET_DIR}...")
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(downloaded, "r") as archive:
        archive.extractall(DATASET_DIR)

    if not dataset_is_ready():
        raise RuntimeError(
            "La extracción terminó, pero no se encontraron las carpetas "
            "images/ y labels/ esperadas."
        )

    if not keep_zip:
        downloaded.unlink(missing_ok=True)
        print("ZIP eliminado para liberar espacio.")

    print(f"Dataset listo en {DATASET_DIR}")
    return DATASET_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Conserva el ZIP descargado después de extraerlo.",
    )
    args = parser.parse_args()
    download_dataset(keep_zip=args.keep_zip)


if __name__ == "__main__":
    main()
