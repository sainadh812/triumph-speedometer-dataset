"""Task models: Colab CUDA training, with explicit Colab CPU inference only."""
from __future__ import annotations

import importlib.util
import os

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ARCHITECTURE_ID = 'dashboard_v1_mobilenetv3small_stride4_lcd4_rpmroi4_spatial_bbox4_mask_v3'
POINT_COUNTS = {'localizer': 8, 'speed': 12, 'rpm': 8}


def require_colab_device(device='cuda', *, inference_only=False):
    """Resolve an explicit Colab device; never fall back from CUDA to CPU."""
    if device not in ('cuda', 'cpu'):
        raise ValueError("device must be 'cuda' or 'cpu'")
    try:
        in_colab = importlib.util.find_spec('google.colab') is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        in_colab = False
    if not (in_colab or os.environ.get('COLAB_RELEASE_TAG')):
        raise RuntimeError('Neural model construction/training/inference is restricted to Google Colab')
    if device == 'cpu':
        if not inference_only:
            raise RuntimeError('CPU is permitted only for explicit Colab inference; training requires CUDA')
        return torch.device('cpu')
    if not torch.cuda.is_available():
        raise RuntimeError('A Colab CUDA runtime is required; no CPU/local model fallback')
    return torch.device('cuda')


def require_colab_cuda():
    """Unchanged training policy: a Colab CUDA runtime is required."""
    return require_colab_device('cuda')


def _validate_execution(images, input_hw, device, inference_only, training):
    if images.device.type != device or images.ndim != 4 or tuple(images.shape[1:]) != (3, *input_hw):
        raise ValueError(f'Expected normalized Colab {device.upper()} N,3,H,W inputs at declared shape')
    if device == 'cpu' and not inference_only:
        raise RuntimeError('CPU training is not permitted')
    if inference_only and (training or torch.is_grad_enabled() or not torch.is_inference_mode_enabled()):
        raise RuntimeError('Inference-only models require eval() and torch.inference_mode()')


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outputs), nn.SiLU(inplace=True))


class InstrumentModel(nn.Module):
    def __init__(self, task, input_hw, pretrained=True, *, device='cuda', inference_only=False):
        require_colab_device(device, inference_only=inference_only)
        super().__init__()
        if task not in POINT_COUNTS:
            raise ValueError('Unknown task')
        self.task, self.input_hw = task, tuple(input_hw)
        # Plain policy attributes add no parameters/buffers or checkpoint keys.
        self._execution_device, self._inference_only = device, bool(inference_only)
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = mobilenet_v3_small(weights=weights).features
        self.top = block(576, 96)
        self.decode16 = block(96 + 48, 64)
        self.decode8 = block(64 + 24, 48)
        self.decode4 = block(48 + 16, 32)
        self.heatmap_head = nn.Conv2d(32, POINT_COUNTS[task], 1)
        self.visibility_head = nn.Linear(576, POINT_COUNTS[task])
        if task == 'localizer':
            self.mask_head = nn.Conv2d(32, 1, 1)
            self.dashboard_heatmap_head = nn.Conv2d(32, 4, 1)
            self.presence_head = nn.Linear(576, 1)

    def forward(self, images):
        _validate_execution(images, self.input_hw, self._execution_device, self._inference_only, self.training)
        value, skips = images, {}
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in (1, 3, 8):
                skips[index] = value
        pooled = value.mean(dim=(-1, -2))
        value = self.top(value)
        for index, decoder in ((8, self.decode16), (3, self.decode8), (1, self.decode4)):
            value = F.interpolate(value, size=skips[index].shape[-2:], mode='bilinear', align_corners=False)
            value = decoder(torch.cat([value, skips[index]], dim=1))
        result = {'heatmap_logits': self.heatmap_head(value), 'visibility_logits': self.visibility_head(pooled)}
        if self.task == 'localizer':
            # Retain spatial location: four dashboard bounding-box corner maps
            # share the dense decoder, rather than regressing position from GAP.
            dashboard_logits = self.dashboard_heatmap_head(value)
            h, w = dashboard_logits.shape[-2:]
            yy, xx = torch.meshgrid(torch.arange(h, device=images.device), torch.arange(w, device=images.device), indexing='ij')
            probability = dashboard_logits.float().flatten(2).softmax(-1)
            # Same pixel-center transform as landmark heatmaps, then normalize
            # by input width/height to match the exported box target convention.
            xs = ((xx.float() + .5) * self.input_hw[1] / w - .5) / self.input_hw[1]
            ys = ((yy.float() + .5) * self.input_hw[0] / h - .5) / self.input_hw[0]
            corners = torch.stack([(probability * xs.flatten()).sum(-1),
                                   (probability * ys.flatten()).sum(-1)], dim=-1)
            result.update(lcd_logits=F.interpolate(self.mask_head(value), size=self.input_hw,
                                                  mode='bilinear', align_corners=False),
                          dashboard_heatmap_logits=dashboard_logits,
                          dashboard_box=torch.cat([corners.amin(dim=1), corners.amax(dim=1)], dim=1),
                          presence_logits=self.presence_head(pooled).squeeze(1))
        return result


def visibility_supervision_loss(logits, visible, visibility_known=None):
    """BCE on known visibility labels only; no guessed negative for unknowns."""
    visible = visible.float()
    known = torch.ones_like(visible) if visibility_known is None else visibility_known.float()
    if known.shape != visible.shape or logits.shape != visible.shape:
        raise ValueError('Visibility supervision shapes differ')
    if bool(torch.any((known != 0) & (known != 1))):
        raise ValueError('visibility_known must contain only zero or one')
    if bool(torch.any((visible > 0) & (known == 0))):
        raise ValueError('A visible landmark must have known visibility')
    per_point = F.binary_cross_entropy_with_logits(logits.float(), visible, reduction='none')
    return (per_point * known).sum() / known.sum().clamp_min(1.)


def training_loss(output, batch, input_hw, sigma=1.5):
    """Spatial CE on visible points; no guessed location loss for absent points."""
    logits = output['heatmap_logits'].float()
    height, width = logits.shape[-2:]
    target_xy = (batch['points'].float() + .5) * logits.new_tensor([width / input_hw[1], height / input_hw[0]]) - .5
    yy, xx = torch.meshgrid(torch.arange(height, device=logits.device), torch.arange(width, device=logits.device), indexing='ij')
    distance = (xx - target_xy[..., 0, None, None]).square() + (yy - target_xy[..., 1, None, None]).square()
    target = torch.exp(-distance / (2 * sigma * sigma))
    target = target / target.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
    per_point = -(target.flatten(2) * logits.flatten(2).log_softmax(-1)).sum(-1)
    visible = batch['visible'].float()
    heatmap_loss = (per_point * visible).sum() / visible.sum().clamp_min(1.)
    visibility_loss = visibility_supervision_loss(output['visibility_logits'], visible, batch.get('visibility_known'))
    pieces = {'heatmap': heatmap_loss, 'visibility': visibility_loss}
    loss = heatmap_loss + .25 * visibility_loss
    if 'lcd_logits' in output:
        mask_logits, truth = output['lcd_logits'].float(), batch['mask'].float()
        # Dice and positive-pixel weighting prevent small LCDs being drowned by background.
        # Empty-target images remain useful BCE negatives; no fabricated foreground target.
        mask_bce = F.binary_cross_entropy_with_logits(mask_logits, truth, pos_weight=mask_logits.new_tensor(3.))
        pred = mask_logits.sigmoid()
        dice = 1 - ((2 * (pred * truth).sum((1, 2, 3)) + 1) /
                    (pred.sum((1, 2, 3)) + truth.sum((1, 2, 3)) + 1)).mean()
        present = batch['present'].float()
        box = batch['box'].float()
        corner_targets = torch.stack([box[:, [0, 1]], box[:, [2, 1]],
                                      box[:, [2, 3]], box[:, [0, 3]]], dim=1)
        corner_targets = corner_targets * logits.new_tensor([input_hw[1], input_hw[0]])
        corner_targets = torch.maximum(corner_targets, corner_targets.new_zeros(()))
        corner_targets = torch.minimum(corner_targets, corner_targets.new_tensor([input_hw[1] - 1., input_hw[0] - 1.]))
        corner_targets = (corner_targets + .5) * logits.new_tensor([width / input_hw[1], height / input_hw[0]]) - .5
        box_distance = (xx - corner_targets[..., 0, None, None]).square() + (yy - corner_targets[..., 1, None, None]).square()
        box_target = torch.exp(-box_distance / (2 * sigma * sigma))
        box_target = box_target / box_target.sum(dim=(-1, -2), keepdim=True).clamp_min(1e-12)
        box_ce = -(box_target.flatten(2) * output['dashboard_heatmap_logits'].float().flatten(2).log_softmax(-1)).sum(-1).mean(1)
        box_heatmap = (box_ce * present).sum() / present.sum().clamp_min(1.)
        per_box = F.smooth_l1_loss(output['dashboard_box'].float(), batch['box'].float(), reduction='none').mean(1)
        bbox = (per_box * present).sum() / present.sum().clamp_min(1.)
        presence = F.binary_cross_entropy_with_logits(output['presence_logits'].float(), present)
        pieces.update(mask_bce=mask_bce, mask_dice=dice, bbox_heatmap=box_heatmap, bbox=bbox, presence=presence)
        loss = loss + mask_bce + dice + box_heatmap + 5 * bbox + .5 * presence
    pieces['total'] = loss
    return loss, pieces
