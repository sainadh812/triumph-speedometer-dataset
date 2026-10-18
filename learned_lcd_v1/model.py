"""A separate unmasked-frame LCD segmenter; the speed model stays frozen."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

PREPROCESSING_ID = 'gray512_fullframe_letterbox_unmasked_lcd_seg_v1_imagenet_normalized'
ARCHITECTURE_ID = 'mobilenetv3small_skip_decoder_stride2_lcd1_v1'
INPUT_SIZE = 512
PRETRAINED_ENUM = 'MobileNet_V3_Small_Weights.IMAGENET1K_V1'


def require_colab_cuda(device='cuda'):
    try:
        colab = importlib.util.find_spec('google.colab') is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        colab = False
    if not (colab or os.environ.get('COLAB_RELEASE_TAG')) or device not in ('cuda','auto') or not torch.cuda.is_available():
        raise RuntimeError('LCD model execution requires Google Colab CUDA; no local/CPU fallback')
    return torch.device('cuda')


def normalize_gray(image):
    array = np.asarray(image)
    if array.shape != (INPUT_SIZE, INPUT_SIZE) or array.dtype != np.uint8:
        raise ValueError('Expected an unmasked 512x512 uint8 grayscale image')
    tensor = torch.from_numpy(array.copy()).float().unsqueeze(0).repeat(3, 1, 1) / 255.
    return (tensor-tensor.new_tensor([.485,.456,.406])[:,None,None])/tensor.new_tensor([.229,.224,.225])[:,None,None]


def _block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
                         nn.GroupNorm(8, outputs), nn.SiLU(inplace=True))


class LCDSegmenter(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        if weights not in (None, 'imagenet', 'IMAGENET1K_V1'):
            raise ValueError('Use explicit imagenet initialization or weights=None')
        selected = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if weights else None
        self.features = mobilenet_v3_small(weights=selected, progress=True).features
        self.lateral = _block(576, 96)
        self.decode16 = _block(96+48, 64)
        self.decode8 = _block(64+24, 32)
        self.decode4 = _block(32+16, 16)
        self.decode2 = _block(16+16, 16)
        self.lcd_head = nn.Conv2d(16, 1, 1)

    def forward(self, image):
        if image.ndim != 4 or image.shape[1:] != (3, INPUT_SIZE, INPUT_SIZE):
            raise ValueError('LCDSegmenter expects N x 3 x 512 x 512 normalized inputs')
        value, skips = image, {}
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in (0,1,3,8):
                skips[index] = value
        value = self.lateral(value)
        for index, block in ((8,self.decode16),(3,self.decode8),(1,self.decode4),(0,self.decode2)):
            value = F.interpolate(value, size=skips[index].shape[-2:], mode='bilinear', align_corners=False)
            value = block(torch.cat([value, skips[index]], dim=1))
        logits = F.interpolate(self.lcd_head(value), size=image.shape[-2:], mode='bilinear', align_corners=False)
        return {'lcd_logits': logits}


def pretrained_identity():
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    path = Path(torch.hub.get_dir())/'checkpoints'/Path(urlparse(weights.url).path).name
    if not path.is_file():
        raise FileNotFoundError('ImageNet weights are not downloaded; no random-weight fallback')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):
            h.update(block)
    return {'mode':'imagenet', 'weight_enum':PRETRAINED_ENUM, 'weight_url':weights.url,
            'weight_file_sha256':h.hexdigest(), 'decoder_initialization':'random', 'optimizer_restored':False}


def configure_stage(model, epoch, warmup_epochs, *, fresh=False):
    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    for block in model.features:
        block.eval()
        for parameter in block.parameters():
            parameter.requires_grad_(False)
    if epoch > warmup_epochs:
        for block in (list(model.features) if fresh else list(model.features)[-4:]):
            block.train()
            for parameter in block.parameters():
                parameter.requires_grad_(True)
    return 'decoder_warmup_encoder_bn_frozen' if epoch <= warmup_epochs else 'late_encoder_finetune'


def segmentation_loss(logits, target, valid, protected=None, protected_available=None, *, analog_weight=.5):
    """Every sample contributes; empty valid regions are errors, not zero loss."""
    logits, target, valid = logits.float(), target.float(), valid.float()
    if logits.shape != target.shape or logits.shape != valid.shape or logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError('Expected equally shaped N x 1 x H x W logits/masks')
    denom = valid.sum((1,2,3))
    if torch.any(denom <= 0):
        raise ValueError('Sample has no valid sensor pixels')
    probability = logits.sigmoid()
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none')*valid).sum((1,2,3))/denom
    intersection = (probability*target*valid).sum((1,2,3))
    dice_loss = 1-(2*intersection+1)/((probability*valid).sum((1,2,3))+(target*valid).sum((1,2,3))+1)
    analog = logits.sum()*0.
    if protected is not None and protected_available is not None:
        mask = protected.float()*valid*(1-target)
        area = mask.sum((1,2,3))
        available = protected_available.bool().reshape(-1) & (area > 0)
        if available.any():
            analog = ((F.softplus(logits)*mask).sum((1,2,3))/area.clamp_min(1))[available].mean()
    total = .5*bce.mean()+.5*dice_loss.mean()+analog_weight*analog
    return total, {'bce':float(bce.mean().detach()), 'dice_loss':float(dice_loss.mean().detach()),
                   'analog_false_mask_loss':float(analog.detach())}
