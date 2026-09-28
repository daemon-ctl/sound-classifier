from collections.abc import Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .preprocessing import CLASS_NAMES


def evaluate_model(model: torch.nn.Module, batches: Iterable, device: torch.device) -> dict:
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    with torch.inference_mode():
        for features, labels in batches:
            logits = model(features.to(device, non_blocking=True))
            targets.extend(labels.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())

    label_ids = list(range(len(CLASS_NAMES)))
    return {
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, labels=label_ids, average="macro", zero_division=0),
        "per_class": classification_report(
            targets,
            predictions,
            labels=label_ids,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=label_ids),
        "targets": np.asarray(targets),
        "predictions": np.asarray(predictions),
    }
