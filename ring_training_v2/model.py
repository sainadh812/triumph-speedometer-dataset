"""Dedicated tick decoder with an independently preserved needle branch.

Transfer uses ring-v1 epoch33 for the trainable encoder/arc/direct-speed branch
and legacy epoch19 for a separate, immutable needle reader. Fresh construction
does not freeze random needle weights. Exact resume must restore freeze_needle
from the experiment's saved initialization policy before configuring its stage.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
from ring_training import ring_model as v1
from training_pilot import train_core as legacy

ARCHITECTURE_ID = "dual_mobilenet_v3_small_frozen_needle_multiscale64_ticks6_arc512_v2"
PREPROCESSING_ID = v1.PREPROCESSING_ID
TICK_NAMES = v1.TICK_NAMES
NEW_PARAMETER_PREFIXES = ("tick_decoder.", "tick_visibility_head.")
DORMANT_PARAMETER_PREFIXES = ("ring_branch.tick_head.", "ring_branch.tick_visibility_head.")
APPROVED_RING_SHA256 = "9dbe35f135b91823a6a65c201c6d3ec557aa11c87a23ef4a335657de6c4720c9"
APPROVED_LEGACY_SHA256 = "ba47ae2a3c07fd9a490342eebb8a01ac160dfe5e9dbca24dd13014ed68cac270"


def _context(in_channels, out_channels=64):
    return nn.Sequential(legacy._conv_block(in_channels, out_channels),
                         legacy._conv_block(out_channels, out_channels))


class TickDecoder(nn.Module):
    """Own spatial decoder; no needle-decoder features or ground-truth inputs."""
    def __init__(self):
        super().__init__()
        self.lateral = nn.Sequential(nn.Conv2d(576, 64, 1, bias=False),
                                     nn.GroupNorm(8, 64), nn.SiLU())
        self.decode16 = _context(64 + 48)
        self.decode8 = _context(64 + 24)
        self.decode4 = _context(64 + 16)
        self.heatmap_head = nn.Conv2d(64, len(TICK_NAMES), 1)

    def forward(self, deep, skips):
        value = self.lateral(deep)
        for index, block in ((8, self.decode16), (3, self.decode8), (1, self.decode4)):
            value = F.interpolate(value, size=skips[index].shape[-2:],
                                  mode="bilinear", align_corners=True)
            value = block(torch.cat((value, skips[index]), dim=1))
        return self.heatmap_head(value), value


def parameter_group(name):
    """Stable optimizer groups; None denotes retained, unused v1 tick tensors."""
    if name.startswith(DORMANT_PARAMETER_PREFIXES):
        return None
    if name.startswith(NEW_PARAMETER_PREFIXES):
        return "new"
    if name.startswith("needle_branch."):
        return "needle"
    if name.startswith("ring_branch.features."):
        return "backbone"
    if name.startswith("ring_branch."):
        return "shared"
    raise ValueError(f"Unclassified model parameter: {name}")


class AugmentedRingModel(nn.Module):
    def __init__(self, pretrained=False, freeze_needle=False):
        super().__init__()
        if type(freeze_needle) is not bool:
            raise ValueError("freeze_needle must be an explicit boolean")
        self.ring_branch = v1.RingSpeedometerModel(pretrained=pretrained)
        self.needle_branch = legacy.SpeedometerModel(pretrained=pretrained)
        self.tick_decoder = TickDecoder()
        # Location-weighted local context plus global orientation/context; tick
        # identities retain their own channel order. Detach the pooling weights
        # so visibility loss cannot pull landmark locations away from tick loss.
        self.tick_visibility_head = nn.Sequential(nn.Linear(576 + 64 * 6, 128),
                                                  nn.SiLU(), nn.Linear(128, 6))
        self.freeze_needle = freeze_needle
        self._stage = "finetune"
        self._unfreeze_all_encoder = False
        self.configure_stage("finetune")

    def forward(self, image):
        branch, skips, value = self.ring_branch, {}, image
        for index, layer in enumerate(branch.features):
            value = layer(value)
            if index in (1, 3, 8):
                skips[index] = value
        deep = value
        value = branch.lateral(deep)
        for index, block in ((8, branch.decode16), (3, branch.decode8), (1, branch.decode4)):
            value = F.interpolate(value, size=skips[index].shape[-2:],
                                  mode="bilinear", align_corners=True)
            value = block(torch.cat((value, skips[index]), dim=1))
        ring_logits = branch.heatmap_head(value)
        ring_points = legacy.heatmap_coordinates(ring_logits)
        pooled = F.adaptive_avg_pool2d(deep, 1).flatten(1)
        speed_input = torch.cat((pooled, ring_points.flatten(1),
                                 ring_points[:, 1] - ring_points[:, 0]), dim=1)
        speed = branch.speed_head(speed_input).squeeze(1).sigmoid() * 180.0
        tick_logits, tick_features = self.tick_decoder(deep, skips)
        spatial_weights = tick_logits.float().flatten(2).softmax(-1).detach()
        local = torch.bmm(spatial_weights, tick_features.float().flatten(2).transpose(1, 2)).flatten(1)
        visibility = self.tick_visibility_head(torch.cat((pooled.float(), local), dim=1))
        if self.freeze_needle:
            self.needle_branch.eval()
            with torch.no_grad():
                needle = self.needle_branch(image)
        else:
            needle = self.needle_branch(image)
        return {"speed": speed, "points": needle["points"],
                "heatmap_logits": needle["heatmap_logits"], "legacy_speed": needle["speed"],
                "tick_points": legacy.heatmap_coordinates(tick_logits),
                "tick_heatmap_logits": tick_logits,
                "arc_logits": branch.arc_head(value, image.shape[-2:]),
                "tick_visibility_logits": visibility}

    def configure_stage(self, stage, *, unfreeze_all_encoder=False):
        if stage not in {"new_heads", "finetune"}:
            raise ValueError(f"Unknown augmented-ring training stage: {stage}")
        if type(unfreeze_all_encoder) is not bool or type(self.freeze_needle) is not bool:
            raise ValueError("Encoder and needle policies must be explicit booleans")
        self._stage, self._unfreeze_all_encoder = stage, unfreeze_all_encoder
        for name, parameter in self.named_parameters():
            group = parameter_group(name)
            enabled = group is not None and (stage == "finetune" or group == "new")
            if group == "needle":
                enabled = not self.freeze_needle
            parameter.requires_grad_(enabled)
        for branch in (self.ring_branch, self.needle_branch):
            for parameter in branch.features.parameters():
                parameter.requires_grad_(False)
            if stage == "finetune" and (branch is self.ring_branch or not self.freeze_needle):
                layers = list(branch.features.children())
                for layer in (layers if unfreeze_all_encoder else layers[-4:]):
                    for parameter in layer.parameters():
                        parameter.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # The two branch implementations preserve frozen encoder BN statistics.
        if getattr(self, "freeze_needle", False):
            self.needle_branch.eval()
        return self


def _read_verified(path, expected_sha256):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Unexpected source checkpoint SHA256: {path.name}")
    marker_path = path.with_suffix(".complete.json")
    if marker_path.is_file():
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        size = marker.get("bytes", marker.get("size_bytes"))
        if marker.get("checkpoint") != path.name or marker.get("sha256") != digest or size != path.stat().st_size:
            raise ValueError("Source checkpoint completion marker failed integrity checks")
    return torch.load(path, map_location="cpu", weights_only=True), digest


def _validate_tensors(model, state):
    destination = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(destination):
        raise ValueError("Source tensor names differ from the declared architecture")
    for name, tensor in state.items():
        expected = destination[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.shape != expected.shape
                or tensor.dtype != expected.dtype or not torch.isfinite(tensor).all()):
            raise ValueError(f"Source tensor shape, dtype or finite-value mismatch: {name}")


def load_transfer_checkpoints(model, ring_path, legacy_path, *,
                              expected_ring_sha256=APPROVED_RING_SHA256,
                              expected_legacy_sha256=APPROVED_LEGACY_SHA256):
    """Weights-only two-source migration, with fresh optimizer/RNG/epoch budget."""
    if not isinstance(model, AugmentedRingModel):
        raise TypeError("Transfer requires AugmentedRingModel")
    ring, ring_sha = _read_verified(ring_path, expected_ring_sha256)
    needle, needle_sha = _read_verified(legacy_path, expected_legacy_sha256)
    if (ring.get("format_version") != 1 or ring.get("checkpoint_kind") != "ring_inference_weights"
            or ring.get("architecture") != v1.ARCHITECTURE_ID
            or ring.get("preprocessing") != PREPROCESSING_ID
            or ring.get("config", {}).get("input_size") != 512):
        raise ValueError("Ring source architecture/preprocessing/input definitions differ")
    if (needle.get("format_version") != 1 or needle.get("architecture") != legacy._ARCHITECTURE_ID
            or needle.get("preprocessing") != PREPROCESSING_ID
            or needle.get("config", {}).get("input_size") != 512
            or needle.get("keypoint_order") != ["pivot", "tip"]
            or needle.get("speed_range_kmh") != [0, 180]):
        raise ValueError("Needle source architecture/preprocessing/input/output definitions differ")
    for source in (ring, needle):
        if type(source.get("best_epoch")) is not int or source["best_epoch"] < 0:
            raise ValueError("Invalid source epoch provenance")
    _validate_tensors(model.ring_branch, ring.get("model_state_dict"))
    _validate_tensors(model.needle_branch, needle.get("model_state_dict"))
    # Validate both sources before mutating either branch.
    model.ring_branch.load_state_dict(ring["model_state_dict"], strict=True)
    model.needle_branch.load_state_dict(needle["model_state_dict"], strict=True)
    model.freeze_needle = True
    model.configure_stage(model._stage, unfreeze_all_encoder=model._unfreeze_all_encoder)
    return {"mode": "transfer", "migration_kind": "two_source_arc33_needle19_tick_decoder_v2",
            "source_ring_sha256": ring_sha, "source_ring_epoch": ring["best_epoch"],
            "source_legacy_sha256": needle_sha, "source_legacy_epoch": needle["best_epoch"],
            "source_ring_labels_sha256": ring.get("identity", {}).get("labels_sha256"),
            "source_legacy_labels_sha256": needle.get("labels_sha256"),
            "target_architecture": ARCHITECTURE_ID, "freeze_needle": True,
            "optimizer_restored": False, "full_state_restored": False,
            "epoch_numbering": "New experiment starts at epoch0; source epochs are provenance"}


@dataclass
class RingLossConfig(v1.RingLossConfig):
    tick_heatmap_weight: float = 1.0
    tick_coordinate_weight: float = 5.0
    # Review20 has114 visible and6 hidden labels. A19:1 hidden-label multiplier
    # equalizes their aggregate contribution; do not confuse this with accuracy.
    visibility_negative_weight: float = 19.0


def training_loss(output, batch, loss_config=None):
    config = RingLossConfig() if loss_config is None else loss_config
    if not isinstance(config, RingLossConfig):
        raise TypeError("loss_config must be v2 RingLossConfig")
    if not math.isfinite(config.visibility_negative_weight) or config.visibility_negative_weight <= 0:
        raise ValueError("visibility_negative_weight must be positive and finite")
    total, components = v1.training_loss(output, batch, replace(config, visibility_weight=0.0))
    target = batch["tick_valid"].float()
    elementwise = F.binary_cross_entropy_with_logits(output["tick_visibility_logits"].float(),
                                                     target, reduction="none")
    weights = target + (1 - target) * config.visibility_negative_weight
    visibility = (elementwise * weights).mean()
    total = total + config.visibility_weight * visibility
    components["visibility_loss"] = float(visibility.detach())
    return total, components
