import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from .preprocessing import CLASS_NAMES


@dataclass(frozen=True)
class TrainingConfig:
    max_epochs: int = 100
    patience: int = 15
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(
    model: torch.nn.Module,
    batches: Iterable,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip_norm: float | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []
    sample_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()

    with context:
        for features, labels in batches:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            total_loss += loss.item() * features.size(0)
            sample_count += features.size(0)
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    label_ids = list(range(len(CLASS_NAMES)))
    return {
        "loss": total_loss / sample_count,
        "accuracy": accuracy_score(targets, predictions),
        "macro_f1": f1_score(targets, predictions, labels=label_ids, average="macro", zero_division=0),
    }


def train_model(
    model: torch.nn.Module,
    train_loader: Iterable,
    validation_loader: Iterable,
    checkpoint_path: str | Path,
    device: torch.device,
    config: TrainingConfig = TrainingConfig(),
    gradient_clip_norm: float | None = None,
) -> list[dict[str, float]]:
    """Выполняет один запуск обучения и сохраняет checkpoint, выбранный только по validation-метрикам."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float]] = []
    best_f1 = -float("inf")
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            gradient_clip_norm=gradient_clip_norm,
        )
        validation_metrics = run_epoch(model, validation_loader, device)
        history.append({
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        })

        improved = (
            validation_metrics["macro_f1"] > best_f1
            or (
                np.isclose(validation_metrics["macro_f1"], best_f1)
                and validation_metrics["loss"] < best_loss
            )
        )
        if improved:
            best_f1 = validation_metrics["macro_f1"]
            best_loss = validation_metrics["loss"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return history
