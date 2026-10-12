"""Entrena, ajusta y compara tres modelos de clasificación de mosquitos.

La ejecución predeterminada usa todo el conjunto para validación/prueba y limita
solo la porción de entrenamiento por clase. Esto produce resultados parciales
reproducibles sin ocultar el desbalance del conjunto original.

Uso:
    python -m src.train_models
    python -m src.train_models --max-train-per-class 300 --force-features
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.modeling import (
    IMAGE_SIZE,
    features_from_rows,
    find_dataset_paths,
    load_annotations,
    predict_proba_compatible,
    select_feature_set,
)


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / "data" / "raw" / "mosquitoalert"
FEATURE_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
RESULT_DIR = ROOT / "entregables" / "resultados"
FIGURE_DIR = ROOT / "entregables" / "figuras"
RANDOM_STATE = 42
PALETTE = ["#173A2B", "#2D5A45", "#7B9B83", "#C3913C", "#A64B38", "#66736C"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--max-train-per-class", type=int, default=600)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--validation-size", type=float, default=0.15)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--force-features", action="store_true")
    return parser.parse_args()


def stratified_partitions(
    df: pd.DataFrame,
    test_size: float,
    validation_size: float,
) -> dict[str, pd.DataFrame]:
    train_val, test = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["class_label"],
    )
    relative_validation = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_val,
        test_size=relative_validation,
        random_state=RANDOM_STATE,
        stratify=train_val["class_label"],
    )
    return {"train": train, "validation": validation, "test": test}


def cap_training_rows(train: pd.DataFrame, maximum: int) -> pd.DataFrame:
    pieces = []
    for _, group in train.groupby("class_label", sort=True):
        if len(group) > maximum:
            group = group.sample(maximum, random_state=RANDOM_STATE)
        pieces.append(group)
    return pd.concat(pieces).sample(frac=1.0, random_state=RANDOM_STATE)


def feature_cache_path(image_size: int, maximum: int) -> Path:
    return FEATURE_DIR / f"model_features_{image_size}px_{maximum}max.npz"


def prepare_features(
    partitions: dict[str, pd.DataFrame],
    image_size: int,
    maximum: int,
    force: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    cache = feature_cache_path(image_size, maximum)
    if cache.exists() and not force:
        print(f"Cargando características desde {cache}")
        saved = np.load(cache, allow_pickle=False)
        matrices = {name: saved[f"X_{name}"] for name in partitions}
        labels = {name: saved[f"y_{name}"] for name in partitions}
        return matrices, labels

    matrices: dict[str, np.ndarray] = {}
    labels: dict[str, np.ndarray] = {}
    for name, rows in partitions.items():
        print(f"Extrayendo {name}: {len(rows):,} imágenes")
        matrix, _ = features_from_rows(rows, image_size=image_size)
        matrices[name] = matrix
        labels[name] = rows["class_label"].to_numpy(dtype=str)
    np.savez_compressed(
        cache,
        **{f"X_{name}": value for name, value in matrices.items()},
        **{f"y_{name}": value for name, value in labels.items()},
    )
    return matrices, labels


def candidate_models() -> dict[str, list[tuple[str, str, object]]]:
    return {
        "Regresión logística": [
            (
                "C=0.5",
                "combined",
                Pipeline([
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=1_200,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    )),
                ]),
            ),
            (
                "C=2.0",
                "combined",
                Pipeline([
                    ("scale", StandardScaler()),
                    ("model", LogisticRegression(
                        C=2.0,
                        class_weight="balanced",
                        max_iter=1_200,
                        solver="lbfgs",
                        random_state=RANDOM_STATE,
                    )),
                ]),
            ),
        ],
        "SVM lineal": [
            (
                "C=0.1",
                "combined",
                Pipeline([
                    ("scale", StandardScaler()),
                    ("model", LinearSVC(
                        C=0.1,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        max_iter=8_000,
                    )),
                ]),
            ),
            (
                "C=0.5",
                "combined",
                Pipeline([
                    ("scale", StandardScaler()),
                    ("model", LinearSVC(
                        C=0.5,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        max_iter=8_000,
                    )),
                ]),
            ),
        ],
        "Bosque aleatorio": [
            (
                "300 árboles, hoja=2",
                "color",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
            (
                "500 árboles, hoja=1",
                "color",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ],
    }


def tune_models(
    matrices: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> tuple[dict[str, dict], list[dict]]:
    selected: dict[str, dict] = {}
    tuning_rows: list[dict] = []
    for family, candidates in candidate_models().items():
        best = None
        for config, feature_set, model in candidates:
            train_x = select_feature_set(matrices["train"], feature_set)
            val_x = select_feature_set(matrices["validation"], feature_set)
            started = time.perf_counter()
            model.fit(train_x, labels["train"])
            fit_seconds = time.perf_counter() - started
            prediction = model.predict(val_x)
            macro_f1 = f1_score(labels["validation"], prediction, average="macro")
            balanced = balanced_accuracy_score(labels["validation"], prediction)
            row = {
                "modelo": family,
                "configuracion": config,
                "caracteristicas": feature_set,
                "f1_macro_validacion": macro_f1,
                "exactitud_balanceada_validacion": balanced,
                "segundos_ajuste_validacion": fit_seconds,
            }
            tuning_rows.append(row)
            print(f"{family} | {config}: F1 macro val={macro_f1:.4f}")
            if best is None or macro_f1 > best["f1_macro"]:
                best = {
                    "model": model,
                    "config": config,
                    "feature_set": feature_set,
                    "f1_macro": macro_f1,
                }
        assert best is not None
        selected[family] = best
    return selected, tuning_rows


def evaluate_and_save(
    selected: dict[str, dict],
    matrices: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    partitions: dict[str, pd.DataFrame],
    image_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    classes = np.array(sorted(np.unique(labels["train"])))
    metrics_rows: list[dict] = []
    per_class_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    all_x = np.vstack([matrices["train"], matrices["validation"]])
    all_y = np.concatenate([labels["train"], labels["validation"]])

    for family, choice in selected.items():
        feature_set = choice["feature_set"]
        model = choice["model"]
        fit_x = select_feature_set(all_x, feature_set)
        test_x = select_feature_set(matrices["test"], feature_set)

        started = time.perf_counter()
        model.fit(fit_x, all_y)
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        prediction = model.predict(test_x)
        probabilities = predict_proba_compatible(model, test_x)
        prediction_seconds = time.perf_counter() - started

        slug = {
            "Regresión logística": "regresion_logistica",
            "SVM lineal": "svm_lineal",
            "Bosque aleatorio": "bosque_aleatorio",
        }[family]
        model_path = MODEL_DIR / f"{slug}.joblib"
        bundle = {
            "model": model,
            "display_name": family,
            "config": choice["config"],
            "feature_set": feature_set,
            "image_size": image_size,
            "classes": list(model.classes_),
            "preprocessing": "EXIF -> recorte bbox -> letterbox -> HOG + HSV",
        }
        joblib.dump(bundle, model_path, compress=3)
        artifact_mb = model_path.stat().st_size / (1024 * 1024)

        metrics_rows.append({
            "modelo": family,
            "configuracion": choice["config"],
            "caracteristicas": feature_set,
            "exactitud": accuracy_score(labels["test"], prediction),
            "exactitud_balanceada": balanced_accuracy_score(labels["test"], prediction),
            "precision_macro": precision_score(labels["test"], prediction, average="macro", zero_division=0),
            "recall_macro": recall_score(labels["test"], prediction, average="macro", zero_division=0),
            "f1_macro": f1_score(labels["test"], prediction, average="macro"),
            "f1_ponderado": f1_score(labels["test"], prediction, average="weighted"),
            "segundos_entrenamiento": fit_seconds,
            "ms_por_imagen": prediction_seconds / len(prediction) * 1_000,
            "tamano_modelo_mb": artifact_mb,
        })

        report = classification_report(
            labels["test"], prediction, labels=classes, output_dict=True, zero_division=0
        )
        for class_name in classes:
            per_class_rows.append({
                "modelo": family,
                "clase": class_name,
                "precision": report[class_name]["precision"],
                "recall": report[class_name]["recall"],
                "f1": report[class_name]["f1-score"],
                "soporte": int(report[class_name]["support"]),
            })

        test_rows = partitions["test"].reset_index(drop=True)
        frame = pd.DataFrame({
            "modelo": family,
            "archivo": test_rows["img_fName"],
            "real": labels["test"],
            "prediccion": prediction,
            "correcta": labels["test"] == prediction,
            "confianza": probabilities.max(axis=1),
        })
        for idx, class_name in enumerate(model.classes_):
            frame[f"prob_{class_name}"] = probabilities[:, idx]
        prediction_frames.append(frame)

    metrics = pd.DataFrame(metrics_rows).sort_values("f1_macro", ascending=False)
    per_class = pd.DataFrame(per_class_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    metrics.to_csv(RESULT_DIR / "metricas_modelos.csv", index=False)
    per_class.to_csv(RESULT_DIR / "metricas_por_clase.csv", index=False)
    predictions.to_csv(RESULT_DIR / "predicciones_prueba.csv", index=False)
    return metrics, per_class, predictions


def save_figures(metrics: pd.DataFrame, per_class: pd.DataFrame, predictions: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=0.95)

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
    fig, ax = plt.subplots(figsize=(10, 5.6))
    sns.barplot(data=tidy, x="valor", y="modelo", hue="métrica", palette=PALETTE[:3], ax=ax)
    ax.set(xlim=(0, 1), xlabel="Puntuación en prueba", ylabel="")
    ax.set_title("Comparación de rendimiento por modelo", weight="bold", loc="left")
    ax.legend(title="", loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_comparacion_metricas.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    recall = per_class.pivot(index="clase", columns="modelo", values="recall")
    fig, ax = plt.subplots(figsize=(9, 5.8))
    sns.heatmap(recall, annot=True, fmt=".2f", cmap="YlGn", vmin=0, vmax=1, ax=ax)
    ax.set_title("Recall por clase", weight="bold", loc="left")
    ax.set(xlabel="", ylabel="")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "02_recall_por_clase.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    models = metrics["modelo"].tolist()
    fig, axes = plt.subplots(1, len(models), figsize=(16, 4.8), sharex=True, sharey=True)
    labels = sorted(predictions["real"].unique())
    for ax, model_name in zip(axes, models):
        subset = predictions[predictions["modelo"] == model_name]
        matrix = confusion_matrix(subset["real"], subset["prediccion"], labels=labels, normalize="true")
        sns.heatmap(matrix, cmap="YlGn", vmin=0, vmax=1, cbar=False, ax=ax)
        ax.set_title(model_name, weight="bold")
        ax.set_xticks(np.arange(len(labels)) + 0.5, labels, rotation=70, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(labels)) + 0.5, labels, rotation=0, fontsize=8)
        ax.set_xlabel("Predicción")
    axes[0].set_ylabel("Clase real")
    fig.suptitle("Matrices de confusión normalizadas", weight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "03_matrices_confusion.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    sns.scatterplot(
        data=metrics,
        x="ms_por_imagen",
        y="f1_macro",
        hue="modelo",
        size="tamano_modelo_mb",
        sizes=(120, 520),
        palette=PALETTE[: len(metrics)],
        ax=ax,
    )
    for row in metrics.itertuples():
        ax.annotate(row.modelo, (row.ms_por_imagen, row.f1_macro), xytext=(6, 6), textcoords="offset points")
    ax.set(xlabel="Milisegundos por imagen", ylabel="F1 macro", ylim=(0, 1))
    ax.set_title("Calidad, velocidad y tamaño del modelo", weight="bold", loc="left")
    ax.legend().remove()
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_calidad_velocidad.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    best_name = metrics.iloc[0]["modelo"]
    best = predictions[predictions["modelo"] == best_name]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    sns.histplot(
        data=best,
        x="confianza",
        hue="correcta",
        bins=20,
        multiple="layer",
        palette={True: "#2D5A45", False: "#A64B38"},
        ax=ax,
    )
    ax.set(xlabel="Confianza aproximada", ylabel="Imágenes")
    ax.set_title(f"Confianza y aciertos de {best_name}", weight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_confianza_modelo_seleccionado.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = find_dataset_paths(args.dataset_root)
    df = load_annotations(paths)
    partitions = stratified_partitions(df, args.test_size, args.validation_size)
    partitions["train"] = cap_training_rows(partitions["train"], args.max_train_per_class)

    split_rows = []
    for split, rows in partitions.items():
        for class_name, count in rows["class_label"].value_counts().sort_index().items():
            split_rows.append({"particion": split, "clase": class_name, "imagenes": int(count)})
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(split_rows).to_csv(RESULT_DIR / "distribucion_particiones.csv", index=False)

    matrices, labels = prepare_features(
        partitions,
        args.image_size,
        args.max_train_per_class,
        args.force_features,
    )
    selected, tuning_rows = tune_models(matrices, labels)
    pd.DataFrame(tuning_rows).to_csv(RESULT_DIR / "ajuste_hiperparametros.csv", index=False)
    metrics, per_class, predictions = evaluate_and_save(
        selected, matrices, labels, partitions, args.image_size
    )
    save_figures(metrics, per_class, predictions)

    metadata = {
        "dataset": "AI Mosquito Alert Challenge 2023",
        "registros_totales": int(len(df)),
        "clases": sorted(df["class_label"].unique().tolist()),
        "particiones": {name: int(len(rows)) for name, rows in partitions.items()},
        "maximo_entrenamiento_por_clase": args.max_train_per_class,
        "semilla": RANDOM_STATE,
        "tamano_imagen": args.image_size,
        "cajas_ajustadas_a_limites": int(df["bbox_adjusted"].sum()),
        "modelo_seleccionado": metrics.iloc[0]["modelo"],
        "criterio_seleccion": "F1 macro en prueba después de selección por F1 macro de validación",
        "alcance": "Resultados parciales con descriptores HOG y HSV; no equivalen a una CNN final.",
    }
    (RESULT_DIR / "metadatos_ejecucion.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nResultados de prueba")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nModelos: {MODEL_DIR}")
    print(f"Figuras: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
