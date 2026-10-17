"""Reproducible first speedometer training pilot (PyTorch / Google Colab).

Run from Python:
    result = train_experiment(TrainConfig(data_dir="/content/bw_dataset_500",
                                         output_dir="/content/run_400"))
This evaluates synthetic validation data, never a real-world test set. Input
images already contain camera and appearance augmentation. No additional image
geometry is changed here, so keypoint targets remain aligned with the pixels.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import random
import shutil
import tempfile
import time
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    input_size: int = 512
    batch_size: int = 8
    head_epochs: int = 5
    finetune_epochs: int = 25
    patience: int = 7
    seed: int = 42
    pretrained: bool = True
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    head_lr: float = 1e-3
    finetune_lr: float = 1e-4
    backbone_lr: float = 2e-5
    weight_decay: float = 1e-4
    num_workers: int = 0
    amp: bool = True
    device: str = "auto"
    min_delta_kmh: float = 0.01
    heatmap_weight: float = 0.2
    coordinate_weight: float = 5.0
    heatmap_sigma: float = 1.5


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def _data_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Dataset path is outside data_dir: {relative}")
    return path


def verify_dataset(data_dir):
    """Check every model input and label, plus pose-group split isolation."""
    root = Path(data_dir)
    label_path = root / "labels.json"
    document = json.loads(label_path.read_text(encoding="utf-8"))
    records = document["samples"]
    if len(records) != document["count"]:
        raise ValueError("labels.json sample count disagrees with metadata")
    counts = Counter(row["dataset_split"] for row in records)
    if dict(counts) != document["split_counts"]:
        raise ValueError("Split counts disagree with labels.json metadata")
    if set(counts) != {"train", "validation"}:
        raise ValueError("Expected nonempty train and validation splits")
    ids, paths, hashes, groups = set(), set(), set(), defaultdict(set)
    for row in records:
        sid, relative = row["sample_id"], row["image"]
        if sid in ids or relative in paths:
            raise ValueError(f"Duplicate sample ID or image path: {sid}")
        ids.add(sid)
        paths.add(relative)
        image_path = _data_path(root, relative)
        digest = _sha256(image_path)
        if digest != row["image_sha256"]:
            raise ValueError(f"Image checksum mismatch: {sid}")
        if digest in hashes:
            raise ValueError(f"Duplicate image content: {sid}")
        hashes.add(digest)
        with Image.open(image_path) as im:
            if im.mode != "L" or im.size != (row["width"], row["height"]):
                raise ValueError(f"Unexpected mode or dimensions: {sid}")
        speed = float(row["speed_kmh"])
        if not math.isfinite(speed) or not 0 <= speed <= 180:
            raise ValueError(f"Invalid speed: {sid}")
        for name in ("pivot", "tip"):
            xy = np.asarray(row["needle_position_px"][name], dtype=float)
            norm = np.asarray(row["needle_position_normalized"][name], dtype=float)
            limits = np.array([row["width"] - 1, row["height"] - 1])
            if xy.shape != (2,) or norm.shape != (2,):
                raise ValueError(f"Invalid keypoint shape: {sid}/{name}")
            if not np.isfinite(xy).all() or not np.all((xy >= 0) & (xy <= limits)):
                raise ValueError(f"Out-of-frame keypoint: {sid}/{name}")
            if not np.allclose(xy / limits, norm, atol=1e-6):
                raise ValueError(f"Normalized keypoint mismatch: {sid}/{name}")
        groups[row["group_id"]].add(row["dataset_split"])
    leaking = [gid for gid, splits in groups.items() if len(splits) != 1]
    if leaking:
        raise ValueError(f"Pose groups cross splits: {leaking[:5]}")
    source = document.get("source", {})
    if source.get("file") and source.get("sha256"):
        if _sha256(_data_path(root, source["file"])) != source["sha256"]:
            raise ValueError("Approved source checksum mismatch")
    report = {
        "count": len(records), "split_counts": dict(counts),
        "pose_groups": len(groups), "pose_group_leakage": False,
        "all_image_hashes_verified": True, "labels_sha256": _sha256(label_path),
        "source_sha256": source.get("sha256"),
        "scope": "synthetic validation; no independent real test set",
    }
    return records, report


def select_group_subset(records, limit, seed):
    """Deterministic nested subsets; never divide variants of a pose group.

    Singleton approved examples precede shuffled paired groups. This makes the
    100/250/400 learning-curve subsets nested and exact for this dataset.
    A requested size that would divide a group is rounded down and reported.
    """
    groups = defaultdict(list)
    for row in records:
        groups[row["group_id"]].append(row)
    ordered = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(ordered)
    ordered.sort(key=lambda gid: len(groups[gid]))
    limit = len(records) if limit is None else int(limit)
    if limit <= 0:
        raise ValueError("Subset size must be positive")
    selected = []
    for gid in ordered:
        group = sorted(groups[gid], key=lambda row: row["sample_id"])
        if len(selected) + len(group) > limit:
            break
        selected.extend(group)
    if not selected:
        raise ValueError("Subset limit is smaller than one complete pose group")
    return selected


class DashboardDataset(Dataset):
    def __init__(self, data_dir, records, input_size=512):
        self.root, self.records, self.input_size = Path(data_dir), records, input_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        image = preprocess_image(self.root / row["image"], self.input_size)
        points = torch.tensor([row["needle_position_normalized"][name]
                               for name in ("pivot", "tip")], dtype=torch.float32)
        return {"image": image, "speed": torch.tensor(float(row["speed_kmh"])),
                "points": points, "index": index}


def preprocess_image(image, input_size=512):
    """Accept a PIL image or image path; return normalized 3-channel CHW input.

    Grayscale is repeated into RGB solely to reuse ImageNet-pretrained weights.
    The caller must crop real camera footage to match the training dashboard.
    """
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            gray = opened.convert("L")
    elif isinstance(image, Image.Image):
        gray = image.convert("L")
    else:
        raise TypeError("image must be a PIL image or a file path")
    gray = gray.resize((input_size, input_size), Image.Resampling.BILINEAR)
    array = np.asarray(gray, dtype=np.float32).copy() / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).repeat(3, 1, 1)
    mean = tensor.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = tensor.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (tensor - mean) / std


def _conv_block(in_channels, out_channels):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                         nn.GroupNorm(8, out_channels), nn.SiLU(inplace=True))


def heatmap_coordinates(logits):
    """Spatial soft-argmax, returning normalized (x,y) in pivot/tip order."""
    batch, channels, height, width = logits.shape
    probabilities = logits.float().flatten(2).softmax(-1).view(batch, channels, height, width)
    xs = torch.linspace(0, 1, width, device=logits.device)
    ys = torch.linspace(0, 1, height, device=logits.device)
    x = (probabilities.sum(dim=2) * xs).sum(-1)
    y = (probabilities.sum(dim=3) * ys).sum(-1)
    return torch.stack((x, y), dim=-1)


class SpeedometerModel(nn.Module):
    """MobileNetV3-Small encoder, spatial keypoint decoder, continuous speed head."""
    def __init__(self, pretrained=True):
        super().__init__()
        try:
            encoder = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
        except Exception as exc:
            if pretrained:
                raise RuntimeError("Pretrained MobileNet weights could not be loaded. "
                                   "Enable internet and retry; no random-weight fallback was used.") from exc
            raise
        self.features = encoder.features
        # torchvision MobileNetV3-Small skip channels: 16@1/4, 24@1/8, 48@1/16.
        self.lateral = _conv_block(576, 96)
        self.decode16 = _conv_block(96 + 48, 64)
        self.decode8 = _conv_block(64 + 24, 32)
        self.decode4 = _conv_block(32 + 16, 16)
        self.heatmap_head = nn.Conv2d(16, 2, 1)
        self.speed_head = nn.Sequential(nn.Linear(576 + 6, 128), nn.SiLU(),
                                        nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, image):
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
        points = heatmap_coordinates(logits)
        relative = points[:, 1] - points[:, 0]
        pooled = F.adaptive_avg_pool2d(deep, 1).flatten(1)
        speed_input = torch.cat((pooled, points.flatten(1), relative), dim=1)
        speed = self.speed_head(speed_input).squeeze(1).sigmoid() * 180.0
        return {"speed": speed, "points": points, "heatmap_logits": logits}

    def configure_stage(self, stage):
        if stage not in {"heads", "finetune"}:
            raise ValueError(f"Unknown stage: {stage}")
        for param in self.features.parameters():
            param.requires_grad_(False)
        if stage == "finetune":
            for layer in list(self.features.children())[-4:]:
                for param in layer.parameters():
                    param.requires_grad_(True)
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Small batches should not overwrite pretrained running BN statistics.
        # Affine BN parameters may still train in the final unfrozen blocks.
        for module in self.features.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self


def training_loss(output, batch, config):
    logits = output["heatmap_logits"].float()
    height, width = logits.shape[-2:]
    points = batch["points"].float()
    yy, xx = torch.meshgrid(torch.arange(height, device=logits.device),
                            torch.arange(width, device=logits.device), indexing="ij")
    centers = points * points.new_tensor([width - 1, height - 1])
    squared = ((xx[None, None] - centers[:, :, 0, None, None]) ** 2
               + (yy[None, None] - centers[:, :, 1, None, None]) ** 2)
    target = torch.exp(-squared / (2 * config.heatmap_sigma ** 2)).flatten(2)
    target = target / target.sum(-1, keepdim=True).clamp_min(1e-12)
    heatmap = -(target * logits.flatten(2).log_softmax(-1)).sum(-1).mean()
    coordinate = F.smooth_l1_loss(output["points"], points, beta=0.02)
    # A unit is 10 km/h and the Huber transition is 2 km/h; gradients stay modest.
    speed = F.smooth_l1_loss(output["speed"].float() / 10,
                           batch["speed"].float() / 10, beta=0.2)
    total = speed + config.heatmap_weight * heatmap + config.coordinate_weight * coordinate
    return total, {"speed_loss": float(speed.detach()),
                   "heatmap_loss": float(heatmap.detach()),
                   "coordinate_loss": float(coordinate.detach())}


def _device(config):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if config.device == "auto" else torch.device(config.device)


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _loader(dataset, config, shuffle=False):
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle,
                      num_workers=config.num_workers, pin_memory=_device(config).type == "cuda",
                      generator=generator, drop_last=False)


def metrics_from_rows(rows):
    if not rows:
        raise ValueError("Cannot evaluate an empty dataset")
    errors = np.array([row["error_kmh"] for row in rows], dtype=float)
    absolute = np.abs(errors)
    return {
        "n": len(rows), "pose_groups": len({row["group_id"] for row in rows}),
        "mae_kmh": float(absolute.mean()),
        "rmse_kmh": float(np.sqrt(np.mean(errors ** 2))),
        "p95_absolute_error_kmh": float(np.percentile(absolute, 95)),
        "max_absolute_error_kmh": float(absolute.max()),
        "bias_kmh": float(errors.mean()),
        **{f"within_{bound}_kmh_percent": float((absolute <= bound).mean() * 100)
           for bound in (1, 2, 5)},
        "pivot_mae_px": float(np.mean([row["pivot_error_px"] for row in rows])),
        "tip_mae_px": float(np.mean([row["tip_error_px"] for row in rows])),
        "keypoint_error_definition": "Mean Euclidean distance in original dataset-image pixels",
        "scope": "synthetic validation only; not real-world test accuracy",
    }


@torch.inference_mode()
def evaluate_model(model, loader, device):
    model.eval()
    rows = []
    for batch in loader:
        output = model(batch["image"].to(device, non_blocking=True))
        speeds = output["speed"].float().cpu().numpy()
        points = output["points"].float().cpu().numpy()
        for local, dataset_index in enumerate(batch["index"].tolist()):
            record = loader.dataset.records[dataset_index]
            predicted_speed = float(speeds[local])
            row = {"sample_id": record["sample_id"], "group_id": record["group_id"],
                   "parent_sample_id": record.get("parent_sample_id", record["sample_id"]),
                   "rotation_degrees": float(record.get("rotation_degrees", 0.0)),
                   "image": record["image"], "dataset_split": record["dataset_split"],
                   "profile": record.get("augmentation", {}).get("photometric", {}).get("profile", "unknown"),
                   "inferred_60_influence": bool(record.get("inferred_60_influence", False)),
                   "target_speed_kmh": float(record["speed_kmh"]),
                   "predicted_speed_kmh": predicted_speed,
                   "error_kmh": predicted_speed - float(record["speed_kmh"])}
            row["absolute_error_kmh"] = abs(row["error_kmh"])
            for index, name in enumerate(("pivot", "tip")):
                predicted = points[local, index] * np.array([record["width"] - 1, record["height"] - 1])
                target = np.array(record["needle_position_px"][name])
                row[f"{name}_error_px"] = float(np.linalg.norm(predicted - target))
                for dimension, axis in enumerate(("x", "y")):
                    row[f"predicted_{name}_{axis}"] = float(predicted[dimension])
                    row[f"target_{name}_{axis}"] = float(target[dimension])
            rows.append(row)
    return metrics_from_rows(rows), rows


def rotation_metrics_from_rows(rows):
    """Report added camera rotation in clockwise image degrees, never needle angle.

    Repeated rotations of one parent/pose remain correlated validation examples;
    per-angle metrics are diagnostics, not independent real-world test results.
    """
    angles = sorted({float(row.get("rotation_degrees", 0.0)) for row in rows})
    by_angle = {
        f"{angle:g}": metrics_from_rows([row for row in rows
                                        if float(row.get("rotation_degrees", 0.0)) == angle])
        for angle in angles
    }
    by_group = {}
    for name, rotated in (("original_zero", False), ("rotated", True)):
        members = [row for row in rows if (float(row.get("rotation_degrees", 0.0)) != 0.0) == rotated]
        if members:
            by_group[name] = metrics_from_rows(members)
    return {"metrics_by_rotation_degrees": by_angle, "metrics_by_rotation_group": by_group,
            "rotation_convention": "Additional whole-image rotation, positive clockwise; not needle angle"}


def _save_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_evaluation_artifacts(rows, history, data_dir, output_dir):
    """Save inspectable predictions, hardest examples and training curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(output_dir)
    _save_csv(out / "validation_predictions.csv", rows)
    _save_csv(out / "history.csv", history)
    hardest = sorted(rows, key=lambda row: row["absolute_error_kmh"], reverse=True)[:12]
    plot_rows = max(1, (len(hardest) + 3) // 4)
    figure, axes = plt.subplots(plot_rows, 4, figsize=(16, 4.2 * plot_rows + 0.6), squeeze=False)
    for axis in axes.flat:
        axis.axis("off")
    for axis, row in zip(axes.flat, hardest):
        with Image.open(Path(data_dir) / row["image"]) as im:
            axis.imshow(im, cmap="gray", vmin=0, vmax=255)
        for prefix, color, style in (("target", "lime", "-"), ("predicted", "magenta", "--")):
            axis.plot([row[f"{prefix}_pivot_x"], row[f"{prefix}_tip_x"]],
                      [row[f"{prefix}_pivot_y"], row[f"{prefix}_tip_y"]],
                      style, color=color, linewidth=1.5, marker="o", markersize=3)
        axis.set_title(f"{row['sample_id']} | target {row['target_speed_kmh']:.1f}\n"
                       f"pred {row['predicted_speed_kmh']:.1f} | error {row['error_kmh']:+.1f} km/h", fontsize=10)
    figure.suptitle("Hardest synthetic validation samples — green: label; magenta: prediction")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out / "worst_predictions.png", dpi=130)
    plt.close(figure)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([h["epoch"] for h in history], [h["train_loss"] for h in history], label="Training loss")
    axes[0].set(xlabel="Epoch", ylabel="Combined training loss")
    axes[1].plot([h["epoch"] for h in history], [h["validation_mae_kmh"] for h in history])
    axes[1].set(xlabel="Epoch", ylabel="Synthetic validation MAE (km/h)")
    axes[1].ticklabel_format(axis="y", style="plain", useOffset=False)
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(out / "learning_curves.png", dpi=140)
    plt.close(figure)


def _optimizer(model, config, stage):
    heads = [p for name, p in model.named_parameters() if p.requires_grad and not name.startswith("features.")]
    groups = [{"params": heads, "lr": config.head_lr if stage == "heads" else config.finetune_lr}]
    backbone = [p for p in model.features.parameters() if p.requires_grad]
    if backbone:
        groups.append({"params": backbone, "lr": config.backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=config.weight_decay)


def train_experiment(config: TrainConfig):
    """Train once, restore the best validation-MAE checkpoint and export reports.

    max_*_samples and pretrained=False support CPU smoke tests; their metrics
    must not be presented as trained model quality. Full runs use all 400/100.
    """
    if config.input_size < 64 or config.input_size % 32:
        raise ValueError("input_size must be >= 64 and divisible by 32 (512 for the pilot)")
    if min(config.head_epochs, config.finetune_epochs) < 0 or config.head_epochs + config.finetune_epochs < 1:
        raise ValueError("At least one training epoch is required")
    if config.batch_size < 1 or config.patience < 1 or config.heatmap_sigma <= 0:
        raise ValueError("batch_size, patience and heatmap_sigma must be positive")
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "best_model.pt").exists():
        raise FileExistsError("Use a new output_dir to preserve the previous experiment")
    _seed_everything(config.seed)
    records, integrity = verify_dataset(config.data_dir)
    train_records = select_group_subset([r for r in records if r["dataset_split"] == "train"],
                                         config.max_train_samples, config.seed)
    val_records = select_group_subset([r for r in records if r["dataset_split"] == "validation"],
                                       config.max_val_samples, config.seed)
    integrity.update({"used_train_samples": len(train_records), "used_validation_samples": len(val_records),
                      "used_train_ids": [r["sample_id"] for r in train_records],
                      "used_validation_ids": [r["sample_id"] for r in val_records]})
    _json_write(out / "data_integrity.json", integrity)
    _json_write(out / "config.json", asdict(config))
    train_loader = _loader(DashboardDataset(config.data_dir, train_records, config.input_size), config, True)
    val_loader = _loader(DashboardDataset(config.data_dir, val_records, config.input_size), config)
    device = _device(config)
    model = SpeedometerModel(pretrained=config.pretrained).to(device)
    amp_enabled = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best_mae, best_epoch, epoch, history = float("inf"), 0, 0, []
    start = time.time()
    best_path = out / "best_model.pt"
    print(f"Device: {device}; train={len(train_records)}, synthetic validation={len(val_records)}", flush=True)
    for stage, epochs in (("heads", config.head_epochs), ("finetune", config.finetune_epochs)):
        if not epochs:
            continue
        # Start fine-tuning from the best head-training checkpoint.
        if stage == "finetune" and best_path.exists():
            model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["model_state_dict"])
        model.configure_stage(stage)
        optimizer = _optimizer(model, config, stage)
        stale = 0
        stage_best = float("inf")
        for _ in range(epochs):
            epoch += 1
            model.train()
            totals = Counter()
            seen = 0
            for batch in train_loader:
                batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                         for key, value in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    output = model(batch["image"])
                    loss, components = training_loss(output, batch, config)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss; no result was silently accepted")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
                scaler.step(optimizer)
                scaler.update()
                count = len(batch["speed"])
                seen += count
                totals["train_loss"] += float(loss.detach()) * count
                for name, value in components.items():
                    totals[name] += value * count
            metrics, _ = evaluate_model(model, val_loader, device)
            mae = metrics["mae_kmh"]
            if not math.isfinite(mae):
                raise FloatingPointError("Nonfinite validation MAE")
            history.append({"epoch": epoch, "stage": stage,
                            **{name: value / seen for name, value in totals.items()},
                            "validation_mae_kmh": mae,
                            "validation_within_2_kmh_percent": metrics["within_2_kmh_percent"],
                            "validation_tip_mae_px": metrics["tip_mae_px"]})
            print(f"Epoch {epoch:02d} {stage:8s} loss={totals['train_loss']/seen:.4f} "
                  f"val MAE={mae:.3f} km/h; within2={metrics['within_2_kmh_percent']:.1f}%", flush=True)
            # Save the actual minimum, separately from the patience improvement threshold.
            if mae < best_mae:
                best_mae, best_epoch = mae, epoch
                torch.save({"format_version": 1, "model_state_dict": model.state_dict(),
                            "config": asdict(config), "best_epoch": best_epoch,
                            "validation_metrics": metrics, "labels_sha256": integrity["labels_sha256"],
                            "keypoint_order": ["pivot", "tip"], "speed_range_kmh": [0, 180],
                            "preprocessing": "PIL grayscale -> square bilinear resize -> repeated RGB -> ImageNet norm",
                            "scope": "synthetic pilot; real camera validation pending"}, best_path)
            if mae < stage_best - config.min_delta_kmh:
                stage_best, stale = mae, 0
            else:
                stale += 1
            _save_csv(out / "history.csv", history)
            if stale >= config.patience:
                print(f"Early stopping {stage}: no material MAE improvement for {stale} epochs", flush=True)
                break
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True)["model_state_dict"])
    metrics, rows = evaluate_model(model, val_loader, device)
    profiles = {profile: metrics_from_rows([row for row in rows if row["profile"] == profile])
                for profile in sorted({row["profile"] for row in rows})}
    speed_bins = {}
    for lower in range(0, 180, 20):
        members = [row for row in rows if lower <= row["target_speed_kmh"] < lower + 20
                   or (lower == 160 and row["target_speed_kmh"] == 180)]
        if members:
            speed_bins[f"{lower}_to_{lower + 20}_kmh"] = metrics_from_rows(members)
    calibration_groups = {}
    for influenced in (False, True):
        members = [row for row in rows if row["inferred_60_influence"] == influenced]
        if members:
            calibration_groups["inferred_60_region" if influenced else "other_regions"] = metrics_from_rows(members)
    report = {"metrics": metrics, "metrics_by_profile": profiles,
              "metrics_by_speed_bin": speed_bins, "metrics_by_calibration_region": calibration_groups,
              "speed_bin_convention": "20-km/h bins are lower-inclusive, upper-exclusive; final bin includes 180",
              "best_epoch": best_epoch,
              "epochs_run": epoch, "elapsed_seconds": time.time() - start, "device": str(device),
              "pretrained": config.pretrained, "training_samples": len(train_records),
              "validation_samples": len(val_records), "labels_sha256": integrity["labels_sha256"],
              "torch_version": str(torch.__version__), "config": asdict(config),
              "limitation": "Shared synthetic master: these are development validation results, "
                            "not evidence of 98% real-world accuracy. No independent real test was run."}
    _json_write(out / "metrics.json", report)
    save_evaluation_artifacts(rows, history, config.data_dir, out)
    return {**report, "checkpoint_path": str(best_path), "metrics_path": str(out / "metrics.json"),
            "predictions_path": str(out / "validation_predictions.csv"),
            "worst_predictions_path": str(out / "worst_predictions.png"),
            "learning_curves_path": str(out / "learning_curves.png")}


def load_model(checkpoint_path, device="auto"):
    """Load this module's tensor-only checkpoint without downloading weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = SpeedometerModel(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, checkpoint


@torch.inference_mode()
def predict_image(model, image, input_size=512):
    """Predict a matched dashboard crop; coordinates refer to resized input.

    No confidence score or unreadability detector is claimed by this model.
    """
    device = next(model.parameters()).device
    model.eval()
    tensor = preprocess_image(image, input_size).unsqueeze(0).to(device)
    result = model(tensor)
    points = result["points"][0].float().cpu().numpy()
    return {"speed_kmh": float(result["speed"][0]),
            "keypoints_normalized": {name: points[index].tolist() for index, name in enumerate(("pivot", "tip"))},
            "keypoints_input_px": {name: (points[index] * (input_size - 1)).tolist()
                                   for index, name in enumerate(("pivot", "tip"))},
            "input_size": input_size}


# Versioned training checkpoints deliberately do not change the model or pilot API.
_ARCHITECTURE_ID = "mobilenet_v3_small_spatial_decoder_16_24_48_576_speed582_v1"
_PREPROCESSING_ID = "PIL grayscale -> square bilinear resize -> repeated RGB -> ImageNet norm"
_RESUME_CONFIG_KEYS = (
    "input_size", "batch_size", "head_epochs", "finetune_epochs", "patience",
    "seed", "head_lr", "finetune_lr", "backbone_lr", "weight_decay",
    "num_workers", "amp", "min_delta_kmh", "heatmap_weight",
    "coordinate_weight", "heatmap_sigma",
)


def _persistent_data(config):
    records, integrity = verify_dataset(config.data_dir)
    train = select_group_subset([r for r in records if r["dataset_split"] == "train"],
                                config.max_train_samples, config.seed)
    validation = select_group_subset([r for r in records if r["dataset_split"] == "validation"],
                                     config.max_val_samples, config.seed)
    integrity = dict(integrity, used_train_samples=len(train), used_validation_samples=len(validation),
                     used_train_ids=[r["sample_id"] for r in train],
                     used_validation_ids=[r["sample_id"] for r in validation])
    return train, validation, integrity


def _identity(config, integrity):
    values = asdict(config)
    return {
        "architecture": _ARCHITECTURE_ID, "preprocessing": _PREPROCESSING_ID,
        "labels_sha256": integrity["labels_sha256"],
        "train_ids": integrity["used_train_ids"],
        "validation_ids": integrity["used_validation_ids"],
        "keypoint_order": ["pivot", "tip"], "speed_range_kmh": [0, 180],
        "training_config": {key: values[key] for key in _RESUME_CONFIG_KEYS},
        "device_type": _device(config).type,
        "amp_enabled": bool(config.amp and _device(config).type == "cuda"),
        "torch_version": str(torch.__version__),
    }


def checkpoint_identity(config):
    """Derive the strict experiment identity used for full-state resume.

    Paths and initialization choice do not affect an already initialized model.
    Dataset contents, exact subsets, training settings and runtime version do.
    Switching runtime/library versions therefore fails explicitly rather than
    presenting a different training trajectory as an exact continuation.
    """
    _, _, integrity = _persistent_data(config)
    return _identity(config, integrity)


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _capture_rng():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        # NumPy arrays/classes are intentionally absent for weights_only=True.
        "numpy": [str(numpy_state[0]), numpy_state[1].tolist(), int(numpy_state[2]),
                  int(numpy_state[3]), float(numpy_state[4])],
        "torch": torch.get_rng_state().clone(),
        "cuda": [state.cpu().clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available() else [],
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def _restore_rng(state):
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.asarray(np_state[1], dtype=np.uint32),
                         np_state[2], np_state[3], np_state[4]))
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG state requires the same number of CUDA devices")
        torch.cuda.set_rng_state_all(state["cuda"])
    torch.backends.cudnn.deterministic = state["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = state["cudnn_benchmark"]


def _copy_verified(local_path, destination, expected_hash):
    """Copy to an unpublished name, verify the destination bytes, then rename."""
    destination = Path(destination)
    partial = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".partial")
    with Path(local_path).open("rb") as source, partial.open("xb") as target:
        shutil.copyfileobj(source, target, length=1024 * 1024)
        target.flush()
        os.fsync(target.fileno())
    if _sha256(partial) != expected_hash:
        raise IOError(f"Checkpoint copy verification failed: {partial}")
    os.replace(partial, destination)
    if _sha256(destination) != expected_hash:
        raise IOError(f"Checkpoint verification after rename failed: {destination}")


def _save_durable(payload, destination, *, kind, identity, epoch, validation_mae_kmh):
    """Publish completion only after local and destination verification.

    Drive's FUSE mount cannot promise remote transactional fsync semantics. A
    completion sidecar plus a fresh SHA256 check makes partial uploads ineligible
    for discovery; prior epoch files remain independent recovery points.
    """
    destination = Path(destination)
    with tempfile.TemporaryDirectory(prefix="speedometer_checkpoint_") as temporary:
        staged = Path(temporary) / "checkpoint.pt"
        with staged.open("wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # Exercise the same safe loader used for restoration before publication.
        torch.load(staged, map_location="cpu", weights_only=True)
        digest = _sha256(staged)
        _copy_verified(staged, destination, digest)
        manifest = {
            "manifest_version": 1, "kind": kind, "checkpoint": destination.name,
            "sha256": digest, "size_bytes": destination.stat().st_size,
            "epoch": int(epoch), "validation_mae_kmh": float(validation_mae_kmh),
            "created_at_ns": time.time_ns(), "identity": identity,
        }
        sidecar = destination.with_suffix(".complete.json")
        staged_manifest = Path(temporary) / "manifest.json"
        _json_write(staged_manifest, manifest)
        _copy_verified(staged_manifest, sidecar, _sha256(staged_manifest))
    return {**manifest, "checkpoint_path": str(destination), "manifest_path": str(sidecar)}


def _load_verified(candidate):
    path = Path(candidate["checkpoint_path"])
    if path.stat().st_size != candidate["size_bytes"] or _sha256(path) != candidate["sha256"]:
        raise IOError(f"Checkpoint changed or is incomplete: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_state_dict", "optimizer_state_dict", "scaler_state_dict", "rng_state",
                "loader_generator_states", "history", "training_state", "best_state", "identity"}
    if (payload.get("format_version") != 2
            or payload.get("checkpoint_kind") != "full_training_state"
            or not required.issubset(payload)):
        raise ValueError(f"Not a complete training-state checkpoint: {path}")
    if payload["identity"] != candidate["identity"]:
        raise ValueError(f"Manifest/checkpoint identity mismatch: {path}")
    state = payload["training_state"]
    state_keys = {"epoch", "stage", "stage_epoch", "stage_best", "stale", "target_epoch",
                  "head_budget", "finetune_budget", "training_complete", "completion_reason"}
    if not isinstance(state, dict) or not state_keys.issubset(state):
        raise ValueError(f"Incomplete stage/counter state: {path}")
    if state["stage"] not in {"heads", "finetune"}:
        raise ValueError(f"Invalid stage: {path}")
    if not {"epoch", "model_state_dict", "validation_metrics", "provenance"}.issubset(payload["best_state"]):
        raise ValueError(f"Incomplete best-model state: {path}")
    if not {"train", "validation"}.issubset(payload["loader_generator_states"]):
        raise ValueError(f"Incomplete DataLoader RNG state: {path}")
    if not {"python", "numpy", "torch", "cuda", "cudnn_deterministic", "cudnn_benchmark"}.issubset(payload["rng_state"]):
        raise ValueError(f"Incomplete random state: {path}")
    if not {"state", "param_groups"}.issubset(payload["optimizer_state_dict"]):
        raise ValueError(f"Incomplete optimizer state: {path}")
    if state["epoch"] != candidate["epoch"]:
        raise ValueError(f"Manifest/checkpoint epoch mismatch: {path}")
    if not math.isclose(payload["validation_metrics"]["mae_kmh"],
                        candidate["validation_mae_kmh"], rel_tol=1e-12):
        raise ValueError(f"Manifest/checkpoint metric mismatch: {path}")
    return payload


def discover_checkpoints(search_root, *, expected_identity=None):
    """Find hash-verified complete full-state checkpoints, skipping bad saves.

    The returned list is sorted by publication time. A complete.json sidecar
    never bypasses byte verification. Incompatible identities are not candidates.
    An empty result is not permission to silently initialize a new model.
    """
    root = Path(search_root)
    candidates = []
    if not root.exists():
        return candidates
    for sidecar in root.rglob("epoch_*.complete.json"):
        try:
            manifest = json.loads(sidecar.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                continue
            if manifest.get("manifest_version") != 1 or manifest.get("kind") != "full_training_state":
                continue
            if any(type(manifest.get(key)) is not int or manifest[key] < minimum
                   for key, minimum in (("size_bytes", 1), ("epoch", 0), ("created_at_ns", 1))):
                continue
            digest = manifest.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                continue
            if not isinstance(manifest.get("identity"), dict):
                continue
            if expected_identity is not None and manifest.get("identity") != expected_identity:
                continue
            filename = manifest["checkpoint"]
            if Path(filename).name != filename:
                continue
            path = sidecar.parent / filename
            if not path.resolve().is_relative_to(root.resolve()):
                continue
            if not math.isfinite(float(manifest["validation_mae_kmh"])):
                continue
            candidate = {**manifest, "checkpoint_path": str(path), "manifest_path": str(sidecar)}
            _load_verified(candidate)
            candidates.append(candidate)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError, EOFError,
                pickle.UnpicklingError):
            continue
    return sorted(candidates, key=lambda item: (item["created_at_ns"], item["checkpoint_path"]))


def find_checkpoint(search_root, mode, expected_identity=None):
    """Choose the newest save or the lowest-MAE epoch among compatible saves."""
    if mode not in {"resume_latest", "resume_best"}:
        raise ValueError("Checkpoint selection mode must be resume_latest or resume_best")
    candidates = discover_checkpoints(search_root, expected_identity=expected_identity)
    if not candidates:
        raise FileNotFoundError("No complete, checksum-valid, compatible full-state checkpoint was found. "
                                "Choose an explicit warm start/fresh run or repair the checkpoint location; "
                                "training has not silently reset.")
    if mode == "resume_latest":
        return candidates[-1]
    return min(candidates, key=lambda item: (item["validation_mae_kmh"], -item["created_at_ns"]))


def _validate_legacy(payload, config, identity):
    if payload.get("format_version") != 1:
        raise ValueError("warm_start expects a legacy format_version=1 best_model.pt")
    if payload.get("labels_sha256") != identity["labels_sha256"]:
        raise ValueError("Legacy checkpoint dataset labels do not match")
    if payload.get("preprocessing") != _PREPROCESSING_ID:
        raise ValueError("Legacy checkpoint preprocessing does not match")
    if payload.get("keypoint_order") != ["pivot", "tip"] or payload.get("speed_range_kmh") != [0, 180]:
        raise ValueError("Legacy checkpoint output definitions do not match")
    legacy_config = payload["config"]
    for key in ("input_size", "max_train_samples", "max_val_samples", "seed"):
        if legacy_config.get(key) != getattr(config, key):
            raise ValueError(f"Legacy checkpoint differs in {key}; warm start requires the same data selection")
    if payload.get("best_epoch", 0) < 1:
        raise ValueError("Legacy checkpoint has no valid best epoch")
    if not math.isfinite(float(payload["validation_metrics"]["mae_kmh"])):
        raise ValueError("Legacy checkpoint has no finite validation MAE")


def _validate_transfer(payload, config):
    """Authorize a weights-only initialization, not a cross-dataset resume."""
    if payload.get("format_version") != 1:
        raise ValueError("warm_start_transfer expects an exported format_version=1 best_model.pt")
    if payload.get("architecture") != _ARCHITECTURE_ID:
        raise ValueError("Transfer checkpoint architecture identity does not match")
    if payload.get("preprocessing") != _PREPROCESSING_ID:
        raise ValueError("Transfer checkpoint preprocessing does not match")
    if payload.get("keypoint_order") != ["pivot", "tip"] or payload.get("speed_range_kmh") != [0, 180]:
        raise ValueError("Transfer checkpoint output definitions do not match")
    if payload.get("config", {}).get("input_size") != config.input_size:
        raise ValueError("Transfer checkpoint input_size differs from the new experiment")
    if type(payload.get("best_epoch")) is not int or payload["best_epoch"] < 0:
        raise ValueError("Transfer checkpoint has no valid source epoch")
    source_hash = payload.get("labels_sha256")
    if not isinstance(source_hash, str) or len(source_hash) != 64:
        raise ValueError("Transfer checkpoint must identify its source dataset")
    if not math.isfinite(float(payload.get("validation_metrics", {}).get("mae_kmh", float("nan")))):
        raise ValueError("Transfer checkpoint has no finite source-validation MAE")


def _inference_payload(best, config, integrity, resume_metadata):
    return {
        "format_version": 1, "model_state_dict": best["model_state_dict"],
        "config": asdict(config), "best_epoch": best["epoch"],
        "validation_metrics": best["validation_metrics"],
        "labels_sha256": integrity["labels_sha256"], "keypoint_order": ["pivot", "tip"],
        "speed_range_kmh": [0, 180], "preprocessing": _PREPROCESSING_ID,
        "architecture": _ARCHITECTURE_ID, "provenance": best["provenance"],
        "resume_metadata": resume_metadata,
        "scope": "synthetic validation; independent real camera validation pending",
    }


def train_persistent(config: TrainConfig, *, mode="fresh", search_root=None,
                     warm_start_path=None, additional_epochs=20,
                     stop_after_epochs=None, extend_by_epochs=0):
    """Train with immutable, verified per-epoch checkpoints on mounted Drive.

    fresh ignores all checkpoints and uses config.pretrained for initialization.
    warm_start imports legacy weights, explicitly resets the optimizer, and sets
    target_epoch=legacy best_epoch+additional_epochs. resume_latest/resume_best
    restore a full state, including its fixed target. Only extend_by_epochs adds
    a new budget. Every invocation requires a new/empty branch output directory.

    warm_start_transfer permits a new dataset/subset experiment with matching
    model/input/output definitions. It copies weights only, starts epoch0 with
    new optimizer/RNG state, and targets additional_epochs. Source epoch and old
    validation metrics are provenance; a new-validation baseline is evaluated
    before any update and preserved for best-model selection and comparison.

    Resume is at an epoch boundary. Bitwise reproducibility still depends on the
    device and deterministic kernels; state restoration never claims otherwise.
    stop_after_epochs is a test/interruption hook, not an early-stopping policy.
    """
    if mode not in {"fresh", "warm_start", "warm_start_transfer", "resume_latest", "resume_best"}:
        raise ValueError(f"Unknown training mode: {mode}")
    if config.input_size < 64 or config.input_size % 32:
        raise ValueError("input_size must be >=64 and divisible by 32")
    if config.batch_size < 1 or config.patience < 1 or config.heatmap_sigma <= 0:
        raise ValueError("batch_size, patience and heatmap_sigma must be positive")
    if min(config.head_epochs, config.finetune_epochs, extend_by_epochs) < 0:
        raise ValueError("Epoch budgets must be nonnegative")
    if mode == "fresh" and config.head_epochs + config.finetune_epochs < 1:
        raise ValueError("Fresh training needs at least one epoch")
    if mode in {"warm_start", "warm_start_transfer"} and (warm_start_path is None or additional_epochs < 1):
        raise ValueError("Warm starts require a weight checkpoint and positive additional_epochs")
    if stop_after_epochs is not None and stop_after_epochs < 0:
        raise ValueError("stop_after_epochs must be nonnegative")
    if mode == "fresh" and stop_after_epochs == 0:
        raise ValueError("A fresh run needs one completed epoch before it has a checkpoint")
    if extend_by_epochs and mode not in {"resume_latest", "resume_best"}:
        raise ValueError("extend_by_epochs is only for explicit resumed-budget extensions")
    out = Path(config.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Choose a new empty output_dir; earlier checkpoints must be preserved")
    train_records, val_records, integrity = _persistent_data(config)
    identity = _identity(config, integrity)
    candidate, restored = None, None
    if mode.startswith("resume_"):
        if search_root is None:
            raise ValueError("Exact resume requires search_root")
        candidate = find_checkpoint(search_root, mode, identity)
        restored = _load_verified(candidate)
    legacy, transfer = None, None
    if mode == "warm_start":
        legacy = torch.load(warm_start_path, map_location="cpu", weights_only=True)
        _validate_legacy(legacy, config, identity)
    elif mode == "warm_start_transfer":
        transfer = torch.load(warm_start_path, map_location="cpu", weights_only=True)
        _validate_transfer(transfer, config)
    out.mkdir(parents=True, exist_ok=True)
    _json_write(out / "data_integrity.json", integrity)
    _json_write(out / "config.json", asdict(config))
    _seed_everything(config.seed)
    train_loader = _loader(DashboardDataset(config.data_dir, train_records, config.input_size), config, True)
    val_loader = _loader(DashboardDataset(config.data_dir, val_records, config.input_size), config)
    device = _device(config)
    model = SpeedometerModel(pretrained=config.pretrained if mode == "fresh" else False).to(device)
    amp_enabled = bool(config.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    best, history, latest_path, transfer_baseline = None, [], None, None
    metadata = {"mode": mode, "source_checkpoint_path": None, "source_epoch": 0,
                "optimizer_restored": False, "full_state_restored": False,
                "initialization": "ImageNet" if config.pretrained else "random",
                "initial_pretrained": bool(config.pretrained),
                "extend_by_epochs": int(extend_by_epochs)}
    state = {"epoch": 0, "stage": "heads" if config.head_epochs else "finetune",
             "stage_epoch": 0, "stage_best": float("inf"), "stale": 0,
             "head_budget": config.head_epochs, "finetune_budget": config.finetune_epochs,
             "target_epoch": config.head_epochs + config.finetune_epochs,
             "training_complete": False, "completion_reason": None}
    metrics = None
    if legacy is not None:
        model.load_state_dict(legacy["model_state_dict"], strict=True)
        epoch = int(legacy["best_epoch"])
        metrics = legacy["validation_metrics"]
        state.update(epoch=epoch, stage="finetune", target_epoch=epoch + int(additional_epochs),
                     finetune_budget=int(additional_epochs))
        metadata.update(source_checkpoint_path=str(Path(warm_start_path)), source_epoch=epoch,
                        source_sha256=_sha256(warm_start_path), initialization="legacy trained weights",
                        initial_pretrained=bool(legacy["config"]["pretrained"]),
                        limitation="Legacy checkpoint has no optimizer/RNG state. Fine-tuning starts with a new optimizer.")
        best = {"epoch": epoch, "model_state_dict": _cpu_tree(model.state_dict()),
                "validation_metrics": metrics,
                "provenance": {"kind": "legacy_baseline", "checkpoint_path": str(warm_start_path),
                               "sha256": metadata["source_sha256"]}}
    elif transfer is not None:
        model.load_state_dict(transfer["model_state_dict"], strict=True)
        source_epoch = int(transfer["best_epoch"])
        state.update(epoch=0, stage="finetune", target_epoch=int(additional_epochs),
                     head_budget=0, finetune_budget=int(additional_epochs))
        metadata.update(
            source_checkpoint_path=str(Path(warm_start_path)), source_epoch=source_epoch,
            source_sha256=_sha256(warm_start_path), initialization="transferred trained weights",
            initial_pretrained=bool(transfer.get("resume_metadata", {}).get(
                "initial_pretrained", transfer["config"]["pretrained"])),
            source_labels_sha256=transfer["labels_sha256"],
            target_labels_sha256=integrity["labels_sha256"],
            source_validation_metrics=transfer["validation_metrics"],
            epoch_numbering="New experiment starts at0; source_epoch records prior training",
            limitation="Weights-only transfer to a new experiment. Optimizer, RNG, history and early-stopping state reset.")
        print(f"Evaluating source epoch{source_epoch} on the new validation set before any training update...",
              flush=True)
        metrics, baseline_rows = evaluate_model(model, val_loader, device)
        if not math.isfinite(metrics["mae_kmh"]):
            raise FloatingPointError("Nonfinite transfer baseline validation MAE")
        baseline_report = {
            "metrics": metrics, **rotation_metrics_from_rows(baseline_rows),
            "source_epoch": source_epoch, "source_checkpoint_sha256": metadata["source_sha256"],
            "source_labels_sha256": transfer["labels_sha256"],
            "target_labels_sha256": integrity["labels_sha256"],
            "validation_samples": len(val_records), "training_updates_before_evaluation": 0,
        }
        transfer_baseline = {"report": baseline_report, "rows": baseline_rows}
        state["stage_best"] = metrics["mae_kmh"]
        best = {"epoch": 0, "model_state_dict": _cpu_tree(model.state_dict()),
                "validation_metrics": metrics,
                "provenance": {"kind": "transferred_weights_baseline", "source_epoch": source_epoch,
                               "checkpoint_path": str(warm_start_path), "sha256": metadata["source_sha256"],
                               "source_labels_sha256": transfer["labels_sha256"],
                               "evaluated_labels_sha256": integrity["labels_sha256"]}}
    elif restored is not None:
        model.load_state_dict(restored["model_state_dict"], strict=True)
        state, best, history = dict(restored["training_state"]), restored["best_state"], restored["history"]
        metrics = restored["validation_metrics"]
        transfer_baseline = restored.get("transfer_baseline")
        metadata.update(source_checkpoint_path=candidate["checkpoint_path"], source_epoch=state["epoch"],
                        source_sha256=candidate["sha256"], optimizer_restored=True, full_state_restored=True,
                        initial_pretrained=restored.get("resume_metadata", {}).get(
                            "initial_pretrained", bool(restored["config"]["pretrained"])),
                        initialization="full training state", parent_resume_metadata=restored.get("resume_metadata"))
        latest_path = candidate["checkpoint_path"]
        if extend_by_epochs:
            state["target_epoch"] += int(extend_by_epochs)
            state["finetune_budget" if state["finetune_budget"] else "head_budget"] += int(extend_by_epochs)
            state.update(training_complete=False, completion_reason=None, stale=0)
            # Explicit budget extension begins a new fine-tuning patience window.
            state["stage_best"] = metrics["mae_kmh"]
    model.configure_stage(state["stage"])
    optimizer = _optimizer(model, config, state["stage"])
    if restored is not None:
        optimizer.load_state_dict(restored["optimizer_state_dict"])
        scaler.load_state_dict(restored["scaler_state_dict"])
        train_loader.generator.set_state(restored["loader_generator_states"]["train"])
        val_loader.generator.set_state(restored["loader_generator_states"]["validation"])
        _restore_rng(restored["rng_state"])
    _json_write(out / "resume_metadata.json", metadata)
    if transfer_baseline is not None:
        _json_write(out / "baseline_validation_metrics.json", transfer_baseline["report"])
        _save_csv(out / "baseline_validation_predictions.csv", transfer_baseline["rows"])
    best_path = out / "best_model.pt"

    def save_best():
        _save_durable(_inference_payload(best, config, integrity, metadata), best_path,
                      kind="inference_best", identity=identity, epoch=best["epoch"],
                      validation_mae_kmh=best["validation_metrics"]["mae_kmh"])

    def save_epoch(kind="epoch_end"):
        nonlocal latest_path
        filename = f"epoch_{state['epoch']:05d}_{uuid.uuid4().hex[:12]}.pt"
        payload = {
            "format_version": 2, "checkpoint_kind": "full_training_state", "save_reason": kind,
            "model_state_dict": _cpu_tree(model.state_dict()),
            "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
            "scaler_state_dict": _cpu_tree(scaler.state_dict()),
            "rng_state": _capture_rng(),
            "loader_generator_states": {"train": train_loader.generator.get_state().clone(),
                                        "validation": val_loader.generator.get_state().clone()},
            "history": list(history), "training_state": dict(state), "best_state": best,
            "identity": identity, "config": asdict(config), "validation_metrics": metrics,
            "resume_metadata": metadata, "labels_sha256": integrity["labels_sha256"],
            "transfer_baseline": transfer_baseline,
            "best_epoch": best["epoch"], "torch_version": str(torch.__version__),
        }
        saved = _save_durable(payload, out / filename, kind="full_training_state", identity=identity,
                              epoch=state["epoch"], validation_mae_kmh=metrics["mae_kmh"])
        latest_path = saved["checkpoint_path"]
        # Pointers are conveniences only; discovery always validates epoch files.
        _json_write(out / "latest_checkpoint.json", saved)

    if best is not None:
        save_best()  # The parent's best remains available even if this branch never improves.
    if legacy is not None:
        save_epoch("legacy_warm_start_boundary")
    elif transfer is not None:
        save_epoch("weights_transfer_baseline_boundary")
    started, epochs_this_call = time.time(), 0
    print(f"{mode}: device={device}; train={len(train_records)}, validation={len(val_records)}; "
          f"epoch {state['epoch']} -> target {state['target_epoch']}", flush=True)
    while (state["epoch"] < state["target_epoch"] and not state["training_complete"]
           and (stop_after_epochs is None or epochs_this_call < stop_after_epochs)):
        # Transition is performed after the prior boundary was saved. Restoring
        # that boundary performs the same weight/optimizer transition exactly once.
        if state["stage"] == "heads" and state["finetune_budget"] and (
                state["stage_epoch"] >= state["head_budget"] or state["stale"] >= config.patience):
            model.load_state_dict(best["model_state_dict"])
            state.update(stage="finetune", stage_epoch=0, stage_best=float("inf"), stale=0)
            model.configure_stage("finetune")
            optimizer = _optimizer(model, config, "finetune")
        model.train()
        totals, seen = Counter(), 0
        for batch in train_loader:
            batch = {key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                     for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                output = model(batch["image"])
                loss, components = training_loss(output, batch, config)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; prior completed checkpoints remain available")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch["speed"])
            seen += count
            totals["train_loss"] += float(loss.detach()) * count
            for name, value in components.items():
                totals[name] += value * count
        metrics, _ = evaluate_model(model, val_loader, device)
        mae = metrics["mae_kmh"]
        if not math.isfinite(mae):
            raise FloatingPointError("Nonfinite validation MAE")
        state["epoch"] += 1
        state["stage_epoch"] += 1
        epochs_this_call += 1
        history.append({"epoch": state["epoch"], "stage": state["stage"],
                        **{name: value / seen for name, value in totals.items()},
                        "validation_mae_kmh": mae,
                        "validation_within_2_kmh_percent": metrics["within_2_kmh_percent"],
                        "validation_tip_mae_px": metrics["tip_mae_px"]})
        improved = best is None or mae < best["validation_metrics"]["mae_kmh"]
        if improved:
            best = {"epoch": state["epoch"], "model_state_dict": _cpu_tree(model.state_dict()),
                    "validation_metrics": metrics,
                    "provenance": {"kind": "training_epoch", "branch_directory": str(out),
                                   "epoch": state["epoch"]}}
        if mae < state["stage_best"] - config.min_delta_kmh:
            state.update(stage_best=mae, stale=0)
        else:
            state["stale"] += 1
        if state["epoch"] >= state["target_epoch"]:
            state.update(training_complete=True, completion_reason="target_epoch_reached")
        elif ((state["stage"] == "finetune" and state["stage_epoch"] >= state["finetune_budget"])
              or (state["stage"] == "heads" and not state["finetune_budget"]
                  and state["stage_epoch"] >= state["head_budget"])):
            state.update(training_complete=True, completion_reason="stage_budget_reached")
        elif state["stale"] >= config.patience and (state["stage"] == "finetune" or not state["finetune_budget"]):
            state.update(training_complete=True, completion_reason="early_stopping")
        # Full state is durable before summaries/plots or mutable inference views.
        save_epoch()
        if improved:
            save_best()
        _save_csv(out / "history.csv", history)
        print(f"Epoch {state['epoch']:02d} {state['stage']:8s} loss={totals['train_loss']/seen:.4f} "
              f"val MAE={mae:.3f}; within2={metrics['within_2_kmh_percent']:.1f}%; checkpoint saved", flush=True)
        if stop_after_epochs is not None and epochs_this_call >= stop_after_epochs:
            break
    if best is None:
        raise RuntimeError("No model was evaluated; no usable result exists")
    model.load_state_dict(best["model_state_dict"])
    evaluation, rows = evaluate_model(model, val_loader, device)
    profiles = {profile: metrics_from_rows([r for r in rows if r["profile"] == profile])
                for profile in sorted({r["profile"] for r in rows})}
    bins = {}
    for lower in range(0, 180, 20):
        members = [r for r in rows if lower <= r["target_speed_kmh"] < lower + 20
                   or (lower == 160 and r["target_speed_kmh"] == 180)]
        if members:
            bins[f"{lower}_to_{lower + 20}_kmh"] = metrics_from_rows(members)
    regions = {}
    for influenced in (False, True):
        members = [r for r in rows if r["inferred_60_influence"] == influenced]
        if members:
            regions["inferred_60_region" if influenced else "other_regions"] = metrics_from_rows(members)
    report = {
        "metrics": evaluation, "metrics_by_profile": profiles, "metrics_by_speed_bin": bins,
        **rotation_metrics_from_rows(rows),
        "metrics_by_calibration_region": regions, "best_epoch": best["epoch"],
        "epochs_run": state["epoch"], "epochs_this_call": epochs_this_call,
        "target_epoch": state["target_epoch"], "training_complete": state["training_complete"],
        "completion_reason": state["completion_reason"], "elapsed_seconds": time.time() - started,
        "device": str(device), "pretrained": metadata["initial_pretrained"],
        "training_samples": len(train_records), "validation_samples": len(val_records),
        "labels_sha256": integrity["labels_sha256"], "torch_version": str(torch.__version__),
        "config": asdict(config), "resume_metadata": metadata, "best_provenance": best["provenance"],
        "baseline_validation_metrics": transfer_baseline["report"]["metrics"] if transfer_baseline else None,
        "baseline_validation_report": transfer_baseline["report"] if transfer_baseline else None,
        "baseline_predictions_path": str(out / "baseline_validation_predictions.csv") if transfer_baseline else None,
        "baseline_metrics_path": str(out / "baseline_validation_metrics.json") if transfer_baseline else None,
        "latest_checkpoint_path": str(latest_path), "checkpoint_path": str(best_path),
        "metrics_path": str(out / "metrics.json"),
        "predictions_path": str(out / "validation_predictions.csv"),
        "worst_predictions_path": str(out / "worst_predictions.png"),
        "learning_curves_path": str(out / "learning_curves.png"),
        "speed_bin_convention": "20-km/h bins are lower-inclusive, upper-exclusive; final bin includes 180",
        "limitation": "Shared synthetic master: development validation only, not evidence of 98% real accuracy.",
    }
    _json_write(out / "metrics.json", report)
    save_evaluation_artifacts(rows, history, config.data_dir, out)
    return report
