"""Versioned ring extension that preserves the legacy speed/needle branches.

New heads are an explicit weights-only migration into a new experiment. Exact
training-state resume must use this architecture's own checkpoints. Arc loss is
computed at input resolution; the thin foreground is never downsampled away.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import torch
from torch import nn
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from training_pilot import train_core as legacy

ARCHITECTURE_ID = "mobilenet_v3_small_spatial_decoder_16_24_48_576_speed582_ticks6_arc512_visibility6_v1"
PREPROCESSING_ID = legacy._PREPROCESSING_ID
TICK_NAMES = ("tick_0", "tick_40", "tick_80", "tick_120", "tick_160", "tick_180")
NEW_PARAMETER_PREFIXES = ("tick_head.", "arc_head.", "tick_visibility_head.")


class _ArcHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.low_resolution = nn.Sequential(nn.Conv2d(16, 8, 3, padding=1), nn.SiLU())
        self.full_resolution = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.SiLU(),
                                             nn.Conv2d(8, 1, 1))

    def forward(self, features, output_size):
        value = self.low_resolution(features)
        value = F.interpolate(value, size=output_size, mode="bilinear", align_corners=True)
        return self.full_resolution(value)


class RingSpeedometerModel(legacy.SpeedometerModel):
    """Same old tensor names and computation, plus independent new output heads."""
    def __init__(self, pretrained=False):
        super().__init__(pretrained=pretrained)
        self.tick_head = nn.Conv2d(16, len(TICK_NAMES), 1)
        self.arc_head = _ArcHead()
        self.tick_visibility_head = nn.Linear(576, len(TICK_NAMES))
        self._ring_stage = "finetune"

    def forward(self, image):
        # Keep the legacy operation ordering, including its speed-head dropout.
        skips = {}
        value = image
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in (1, 3, 8):
                skips[index] = value
        deep = value
        value = self.lateral(deep)
        for index, block in ((8, self.decode16), (3, self.decode8), (1, self.decode4)):
            value = F.interpolate(value, size=skips[index].shape[-2:],
                                  mode="bilinear", align_corners=True)
            value = block(torch.cat((value, skips[index]), dim=1))
        logits = self.heatmap_head(value)
        points = legacy.heatmap_coordinates(logits)
        relative = points[:, 1] - points[:, 0]
        pooled = F.adaptive_avg_pool2d(deep, 1).flatten(1)
        speed_input = torch.cat((pooled, points.flatten(1), relative), dim=1)
        speed = self.speed_head(speed_input).squeeze(1).sigmoid() * 180.0
        tick_logits = self.tick_head(value)
        return {"speed": speed, "points": points, "heatmap_logits": logits,
                "tick_points": legacy.heatmap_coordinates(tick_logits),
                "tick_heatmap_logits": tick_logits,
                "arc_logits": self.arc_head(value, image.shape[-2:]),
                "tick_visibility_logits": self.tick_visibility_head(pooled)}

    def configure_stage(self, stage, *, unfreeze_all_encoder=False):
        if stage not in {"new_heads", "finetune"}:
            raise ValueError(f"Unknown ring training stage: {stage}")
        if type(unfreeze_all_encoder) is not bool:
            raise ValueError("unfreeze_all_encoder must be an explicit boolean")
        self._ring_stage = stage
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(stage == "finetune" or name.startswith(NEW_PARAMETER_PREFIXES))
        for parameter in self.features.parameters():
            parameter.requires_grad_(False)
        if stage == "finetune":
            # Fresh random initialization cannot keep early random layers fixed.
            # Migrated models retain the original final-four-block policy.
            layers = list(self.features.children())
            for layer in (layers if unfreeze_all_encoder else layers[-4:]):
                for parameter in layer.parameters():
                    parameter.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)  # Legacy policy always freezes encoder BN statistics.
        if getattr(self, "_ring_stage", "finetune") == "new_heads":
            # Freezing parameters alone does not freeze BatchNorm's running state.
            for name, module in self.named_modules():
                if not name.startswith(NEW_PARAMETER_PREFIXES) and isinstance(module, nn.modules.batchnorm._BatchNorm):
                    module.eval()
        return self


def load_legacy_checkpoint(model, path):
    """Validate and copy all old tensors; only the three new heads stay random.

    No optimizer, history, RNG, or epoch budget is restored. The returned source
    epoch is provenance; a trainer starts the expanded experiment at epoch zero.
    A present completion sidecar is also checked before deserialization.
    """
    if not isinstance(model, RingSpeedometerModel):
        raise TypeError("Migration requires RingSpeedometerModel")
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    sidecar = path.with_suffix(".complete.json")
    if sidecar.is_file():
        marker = json.loads(sidecar.read_text(encoding="utf-8"))
        if (marker.get("sha256") != digest or marker.get("size_bytes") != path.stat().st_size
                or marker.get("checkpoint") != path.name or marker.get("kind") != "inference_best"):
            raise ValueError("Legacy checkpoint completion marker failed integrity checks")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (checkpoint.get("format_version") != 1
            or checkpoint.get("architecture") != legacy._ARCHITECTURE_ID
            or checkpoint.get("preprocessing") != PREPROCESSING_ID
            or checkpoint.get("keypoint_order") != ["pivot", "tip"]
            or checkpoint.get("speed_range_kmh") != [0, 180]
            or checkpoint.get("config", {}).get("input_size") != 512):
        raise ValueError("Legacy architecture, input, preprocessing or output definitions differ")
    source_epoch = checkpoint.get("best_epoch")
    labels_sha = checkpoint.get("labels_sha256", "")
    source_mae = checkpoint.get("validation_metrics", {}).get("mae_kmh", float("nan"))
    if (type(source_epoch) is not int or source_epoch < 0 or not isinstance(labels_sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", labels_sha)
            or not isinstance(source_mae, (int, float)) or not math.isfinite(source_mae)):
        raise ValueError("Legacy checkpoint has invalid epoch/dataset/validation provenance")
    destination = model.state_dict()
    old_keys = {name for name in destination if not name.startswith(NEW_PARAMETER_PREFIXES)}
    new_keys = set(destination) - old_keys
    source = checkpoint.get("model_state_dict")
    if not isinstance(source, dict) or set(source) != old_keys:
        raise ValueError("Legacy tensor names must match every existing tensor exactly")
    for name, tensor in source.items():
        expected = destination[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected.shape
                or tensor.dtype != expected.dtype or not torch.isfinite(tensor).all()):
            raise ValueError(f"Legacy tensor shape, dtype, or finite-value check failed: {name}")
    missing, unexpected = model.load_state_dict(source, strict=False)
    if set(missing) != new_keys or unexpected:
        raise RuntimeError("Migration unexpectedly omitted or added tensors")
    return {"mode": "legacy_architecture_migration", "source_sha256": digest,
            "source_epoch": source_epoch, "source_labels_sha256": labels_sha,
            "source_architecture": legacy._ARCHITECTURE_ID, "target_architecture": ARCHITECTURE_ID,
            "source_validation_mae_kmh": float(source_mae),
            "copied_tensor_names": sorted(old_keys), "new_tensor_names": sorted(new_keys),
            "optimizer_restored": False, "full_state_restored": False,
            "epoch_numbering": "Expanded experiment starts at 0; source epoch is provenance"}


@dataclass
class RingLossConfig:
    # Existing speed/needle terms retain exactly the legacy definitions/weights.
    heatmap_weight: float = 0.2
    coordinate_weight: float = 5.0
    heatmap_sigma: float = 1.5
    tick_heatmap_weight: float = 0.2
    tick_coordinate_weight: float = 5.0
    arc_bce_weight: float = 0.5
    arc_dice_weight: float = 0.5
    arc_positive_weight: float = 10.0
    visibility_weight: float = 0.1


def _masked_mean(values, valid):
    return (values * valid).sum() / valid.sum().clamp_min(1.0)


def training_loss(output, batch, loss_config=None):
    """Return scalar loss and detached components with occluded labels excluded.

    Arc visible/valid masks remain at full input resolution. Valid-background
    pixels contribute to the loss; ignored/occluded/border pixels do not. Tick
    visibility is a supervised output, not a calibrated probability of accuracy.
    """
    config = RingLossConfig() if loss_config is None else loss_config
    if not isinstance(config, RingLossConfig):
        raise TypeError("loss_config must be RingLossConfig")
    if (any(not math.isfinite(value) or value < 0 for value in vars(config).values())
            or config.heatmap_sigma <= 0 or config.arc_positive_weight <= 0):
        raise ValueError("Loss weights must be finite/nonnegative and scales positive")
    old_loss, components = legacy.training_loss(output, batch, config)
    logits = output["tick_heatmap_logits"].float()
    target_points = batch["tick_points"].float()
    valid = batch["tick_valid"].float()
    if logits.shape[:2] != valid.shape or target_points.shape != (*valid.shape, 2):
        raise ValueError("Tick target/visibility shapes differ from six heatmaps")
    height, width = logits.shape[-2:]
    yy, xx = torch.meshgrid(torch.arange(height, device=logits.device),
                            torch.arange(width, device=logits.device), indexing="ij")
    centers = target_points * target_points.new_tensor([width - 1, height - 1])
    squared = ((xx[None, None] - centers[:, :, 0, None, None])**2
               + (yy[None, None] - centers[:, :, 1, None, None])**2)
    target = torch.exp(-squared / (2 * config.heatmap_sigma**2)).flatten(2)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
    tick_heatmap = _masked_mean(-(target * logits.flatten(2).log_softmax(-1)).sum(-1), valid)
    tick_coordinate = _masked_mean(F.smooth_l1_loss(output["tick_points"].float(), target_points,
                                                   beta=0.02, reduction="none").mean(-1), valid)
    arc_logits, arc_target, arc_valid = (output["arc_logits"].float(), batch["arc_mask"].float(),
                                        batch["arc_valid"].float())
    if arc_logits.shape != arc_target.shape or arc_target.shape != arc_valid.shape:
        raise ValueError("Arc loss requires matching full-resolution logits/target/valid masks")
    positive_weight = arc_logits.new_tensor(config.arc_positive_weight)
    arc_bce = _masked_mean(F.binary_cross_entropy_with_logits(arc_logits, arc_target,
                                                             pos_weight=positive_weight, reduction="none"), arc_valid)
    probabilities, target_visible = arc_logits.sigmoid() * arc_valid, arc_target * arc_valid
    axes = tuple(range(1, arc_logits.ndim))
    intersection = (probabilities * target_visible).sum(dim=axes)
    denominator = probabilities.sum(dim=axes) + target_visible.sum(dim=axes)
    has_pixels = (arc_valid.sum(dim=axes) > 0).float()
    arc_dice = _masked_mean(1 - (2 * intersection + 1) / (denominator + 1), has_pixels)
    visibility = F.binary_cross_entropy_with_logits(output["tick_visibility_logits"].float(), valid)
    total = (old_loss + config.tick_heatmap_weight * tick_heatmap
             + config.tick_coordinate_weight * tick_coordinate + config.arc_bce_weight * arc_bce
             + config.arc_dice_weight * arc_dice + config.visibility_weight * visibility)
    components.update(tick_heatmap_loss=float(tick_heatmap.detach()),
                      tick_coordinate_loss=float(tick_coordinate.detach()),
                      arc_bce_loss=float(arc_bce.detach()), arc_dice_loss=float(arc_dice.detach()),
                      visibility_loss=float(visibility.detach()))
    return total, components
