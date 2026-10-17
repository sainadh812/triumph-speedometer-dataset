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
import random
import time
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
