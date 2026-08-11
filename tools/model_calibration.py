"""Temperature scaling and calibration metrics for trained AI models."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F


def calibration_path_for_model(model_path) -> Path:
    return Path(model_path).with_suffix(".calibration.json")


def load_encoder_temperature(model_path) -> Tuple[float, Path]:
    path = calibration_path_for_model(model_path)
    if not path.is_file():
        return 1.0, path
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    temperature = float(data["temperature"])
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"Invalid encoder temperature in {path}: {temperature!r}")
    return temperature, path


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and greater than zero")
    return logits / temperature


def expected_calibration_error(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    bins: int = 15,
) -> float:
    if bins < 1:
        raise ValueError("bins must be positive")
    probabilities = torch.softmax(logits, dim=1)
    confidence, predictions = probabilities.max(dim=1)
    correct = predictions.eq(targets)
    error = torch.zeros((), dtype=logits.dtype, device=logits.device)
    boundaries = torch.linspace(0.0, 1.0, bins + 1, device=logits.device)
    for index in range(bins):
        lower, upper = boundaries[index], boundaries[index + 1]
        in_bin = confidence.gt(lower) & confidence.le(upper)
        if in_bin.any():
            fraction = in_bin.float().mean()
            accuracy = correct[in_bin].float().mean()
            mean_confidence = confidence[in_bin].mean()
            error = error + fraction * torch.abs(accuracy - mean_confidence)
    return float(error.item())


def classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    bins: int = 15,
) -> Dict[str, float]:
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets have incompatible shapes")
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(targets, num_classes=logits.shape[1]).to(probabilities.dtype)
    return {
        "accuracy": float(probabilities.argmax(dim=1).eq(targets).float().mean().item()),
        "nll": float(F.cross_entropy(logits, targets).item()),
        "ece": expected_calibration_error(logits, targets, bins=bins),
        "brier": float(torch.mean(torch.sum((probabilities - one_hot) ** 2, dim=1)).item()),
    }


def fit_temperature(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if logits.ndim != 2 or targets.ndim != 1 or logits.shape[0] != targets.shape[0]:
        raise ValueError("logits and targets have incompatible shapes")
    optimization_logits = logits.detach().to(dtype=torch.float64)
    optimization_targets = targets.detach().to(device=optimization_logits.device)
    log_temperature = torch.zeros(
        (), dtype=optimization_logits.dtype, device=optimization_logits.device, requires_grad=True
    )
    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(1e-3, 1e3)
        loss = F.cross_entropy(optimization_logits / temperature, optimization_targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    temperature = float(log_temperature.detach().exp().clamp(1e-3, 1e3).item())
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("Temperature optimization produced an invalid value")
    return temperature
