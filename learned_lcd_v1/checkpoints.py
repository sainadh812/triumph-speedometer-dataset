"""LCD-specific validation around the established atomic Drive publisher."""
from __future__ import annotations
import math
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from real_adaptation_v3 import train_needle as durable

KIND='learned_lcd_full_state_v1'
INFERENCE_KIND='learned_lcd_inference_v1'
FORMAT_VERSION=1
sha256=durable._sha
atomic_json=durable._json
cpu=durable._cpu
rng_state=durable._rng_state
restore_rng=durable._restore_rng
seed=durable._seed


def validate(payload):
    required={'checkpoint_kind','format_version','identity','model_state_dict','optimizer_state_dict',
        'scaler_state_dict','scheduler_state_dict','rng_state','sampler_state','loader_generator_states',
        'epoch','history','best_epoch','best_score','best_checkpoint','bad_epochs','config','initialization','optimization_progress'}
    if not isinstance(payload,dict) or not required.issubset(payload): raise ValueError('Incomplete LCD checkpoint')
    if payload['checkpoint_kind']!=KIND or payload['format_version']!=FORMAT_VERSION: raise ValueError('Wrong checkpoint family')
    epoch=payload['epoch']; history=payload['history']
    if type(epoch) is not int or epoch<0 or [r['epoch'] for r in history]!=list(range(epoch+1)):
        raise ValueError('Incomplete LCD epoch history')
    progress=payload['optimization_progress']
    keys={'optimizer_updates','optimized_training_images','amp_skipped_batches','consecutive_amp_skips','max_consecutive_amp_skips_seen'}
    if not isinstance(progress,dict) or set(progress)!=keys or any(type(v) is not int or v<0 for v in progress.values()):
        raise ValueError('Invalid optimizer/AMP progress')
    if any(row.get('optimizer_updates',0)<=0 for row in history[1:]):
        raise ValueError('Completed epoch has zero optimizer updates')
    for key in ('optimizer_updates','optimized_training_images','amp_skipped_batches'):
        if progress[key]!=sum(row.get(key,0) for row in history):raise ValueError('Optimizer/AMP progress differs from epoch history')
    if progress['consecutive_amp_skips']>progress['max_consecutive_amp_skips_seen']:
        raise ValueError('Invalid consecutive AMP skip count')
    scores=[r['validation']['selection_score'] for r in history]
    if any(not math.isfinite(s) for s in scores): raise ValueError('Nonfinite selection score')
    best=min(range(len(scores)),key=lambda i:scores[i])
    if payload['best_epoch']!=best or payload['best_score']!=scores[best]: raise ValueError('Best epoch differs from history')
    if best<epoch:
        reference=payload['best_checkpoint']
        if (not isinstance(reference,dict) or not isinstance(reference.get('path'),str)
                or reference.get('epoch')!=best or not isinstance(reference.get('sha256'),str)
                or len(reference['sha256'])!=64):
            raise ValueError('Missing earlier best full-state checkpoint reference')
    if payload['scheduler_state_dict'] is not None: raise ValueError('Constant learning rates require no scheduler state')
    if payload['sampler_state']['epoch']!=epoch: raise ValueError('Sampler epoch mismatch')
    if not {'python','numpy','torch','cuda'}.issubset(payload['rng_state']): raise ValueError('Missing RNG state')
    if set(payload['loader_generator_states'])!={'train','val'}: raise ValueError('Missing loader RNG states')
    for value in [payload['rng_state']['torch'],*payload['loader_generator_states'].values()]: torch.Generator().set_state(value)
    if not {'state','param_groups'}.issubset(payload['optimizer_state_dict']): raise ValueError('Missing optimizer state')
    if epoch and not payload['optimizer_state_dict']['state']: raise ValueError('Missing optimizer moments')
    if not payload['model_state_dict'] or any(not torch.is_tensor(t) or not torch.isfinite(t).all() for t in payload['model_state_dict'].values()):
        raise ValueError('Invalid model tensors')


def save(payload,path):
    validate(payload)
    marker=durable._durable_torch_save(payload,Path(path))
    marker.update(epoch=payload['epoch'],identity=payload['identity'],selection_loss=payload['history'][-1]['validation']['selection_score'])
    atomic_json(Path(path).with_suffix('.complete.json'),{k:v for k,v in marker.items() if k!='path'})
    return marker


def load(path,identity=None):
    payload,marker=durable._read_complete(path)
    validate(payload)
    if marker.get('identity')!=payload['identity'] or marker.get('epoch')!=payload['epoch'] or marker.get('selection_loss')!=payload['history'][-1]['validation']['selection_score']:
        raise ValueError('LCD completion metadata mismatch')
    if identity is not None and payload['identity']!=identity: raise ValueError('Incompatible LCD dataset/code/runtime/configuration')
    return payload,marker


def resolve(root,mode,identity,checkpoint=None):
    if mode in ('imagenet','fresh'):
        if checkpoint is not None: raise ValueError('Initialization cannot take a resume checkpoint')
        return None,[]
    if mode=='resume_explicit':
        if not checkpoint: raise ValueError('Explicit resume requires checkpoint')
        payload,marker=load(checkpoint,identity)
        return {'path':str(Path(checkpoint).resolve()),**marker},[]
    if mode not in ('resume_best','resume_latest') or checkpoint is not None: raise ValueError('Invalid resume mode/checkpoint combination')
    accepted,rejected=[],[]
    for sidecar in sorted(Path(root).rglob('epoch_*.complete.json')):
        path=sidecar.with_name(sidecar.name[:-len('.complete.json')]+'.pt')
        try:
            _,marker=load(path,identity)
            accepted.append({**marker,'path':str(path.resolve())})
        except Exception as exc: rejected.append({'path':str(path),'reason':str(exc)})
    if not accepted: raise ValueError('No verified compatible LCD checkpoint; no silent fresh fallback')
    selected=(max(accepted,key=lambda r:r['created_at_ns']) if mode=='resume_latest'
              else min(accepted,key=lambda r:(r['selection_loss'],r['epoch'])))
    return selected,rejected


def load_export(path):
    payload,marker=durable._read_complete(path)
    if payload.get('checkpoint_kind')!=INFERENCE_KIND or payload.get('format_version')!=FORMAT_VERSION:
        raise ValueError('Expected LCD inference export')
    if not payload.get('model_state_dict') or any(not torch.isfinite(t).all() for t in payload['model_state_dict'].values()):
        raise ValueError('Invalid inference weights')
    return payload,marker
