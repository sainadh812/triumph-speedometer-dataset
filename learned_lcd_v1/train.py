"""Independent LCD segmentation experiment; optimization runs only in Colab."""
from __future__ import annotations
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict,dataclass
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import time
import uuid

import cv2
import numpy as np
import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset,DataLoader,Sampler
import torchvision

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from learned_lcd_v1 import checkpoints as cp
from learned_lcd_v1.model import (LCDSegmenter,ARCHITECTURE_ID,PREPROCESSING_ID,normalize_gray,
    segmentation_loss,configure_stage,pretrained_identity,require_colab_cuda)
from learned_lcd_v1.evaluate import evaluate_model


@dataclass
class TrainConfig:
    data_dir:str
    output_dir:str
    epochs:int=20
    batch_size:int=12
    warmup_epochs:int=2
    patience:int=5
    min_delta:float=.001
    decoder_lr:float=3e-4
    encoder_lr:float=1e-5
    weight_decay:float=1e-4
    grad_clip:float=5.
    analog_weight:float=.5
    amp:bool=True
    device:str='cuda'
    num_workers:int=0
    seed:int=42
    progress_every_batches:int=50


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _file(root,relative):
    if not isinstance(relative,str):raise ValueError('Expected relative asset path')
    path=(root/relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():raise ValueError('Invalid dataset path: '+relative)
    return path


def verify_dataset(directory,*,progress=True):
    root=Path(directory).resolve();manifest_path=root/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('preprocessing_id')!=PREPROCESSING_ID or manifest.get('input_size') not in (512,[512,512]):
        raise ValueError('Expected unmasked grayscale full-frame LCD input contract')
    if manifest.get('samples_file')!='samples.jsonl' or manifest.get('source_file')!='sources.json':
        raise ValueError('Unexpected LCD annotation file names')
    hashes=manifest.get('sha256')
    if not isinstance(hashes,dict) or not {'samples.jsonl','sources.json'}.issubset(hashes):
        raise ValueError('Dataset annotations must be checksum protected')
    if progress:print('Verifying LCD dataset file hashes...',flush=True)
    for index,(name,value) in enumerate(hashes.items(),1):
        expected=value.get('sha256') if isinstance(value,dict) else value
        if cp.sha256(_file(root,name))!=expected:raise ValueError('Dataset hash mismatch: '+name)
        if progress and index%500==0:print(f'LCD hashes {index}/{len(hashes)}',flush=True)
    source_document=json.loads((root/'sources.json').read_text(encoding='utf-8'))
    sources=source_document.get('sources') if isinstance(source_document,dict) else source_document
    if not isinstance(sources,list) or not sources:raise ValueError('Missing source provenance')
    source_map={row['source_id']:row for row in sources}
    if len(source_map)!=len(sources):raise ValueError('Duplicate source id')
    for source in sources:
        if source.get('split') not in ('train','val','test') or not source.get('source_group'):
            raise ValueError('Invalid source split/group')
        status=str(source.get('annotation_status','')).lower()
        if 'review' not in status or 'pending' in status or 'excluded' in status:
            raise ValueError('LCD source was not visually reviewed: '+source['source_id'])
        if source.get('model_predictions_used_for_labels') is not False:
            raise ValueError('LCD labels must not come from model predictions')
        for field in ('image','lcd_mask','valid_mask','protected_mask'):
            relative=source.get(field)
            if relative is not None and relative not in hashes:raise ValueError('Unprotected source review asset: '+relative)
    rows={'train':[],'val':[],'test':[]};ids=set();paths=set()
    source_splits={};pixels={split:set() for split in rows};recordings={split:set() for split in rows}
    annotations=[json.loads(line) for line in (root/'samples.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    for index,row in enumerate(annotations,1):
        required={'id','source_id','source_group','split','image','lcd_mask','valid_mask','rotation_deg'}
        if not required.issubset(row) or row['id'] in ids or row['image'] in paths:raise ValueError('Missing/duplicate LCD sample')
        ids.add(row['id']);paths.add(row['image'])
        source=source_map.get(row['source_id']);split=row['split']
        if source is None or split not in rows or split!=source['split'] or row['source_group']!=source['source_group']:
            raise ValueError('Source/sample split or recording provenance mismatch')
        source_splits.setdefault(row['source_id'],set()).add(split);recordings[split].add(row['source_group'])
        if not math.isfinite(float(row['rotation_deg'])):raise ValueError('Nonfinite rotation label')
        arrays={}
        for field in ('image','lcd_mask','valid_mask','protected_mask'):
            relative=row.get(field)
            if relative is None and field=='protected_mask':continue
            if relative not in hashes:raise ValueError('Unprotected image/mask: '+str(relative))
            with Image.open(_file(root,relative)) as image:
                if image.mode!='L' or image.size!=(512,512):raise ValueError('LCD assets must be grayscale512 PNGs')
                arrays[field]=np.asarray(image).copy()
            if field!='image' and not np.isin(arrays[field],[0,255]).all():raise ValueError('Nonbinary LCD/valid/protected mask')
        lcd,valid=arrays['lcd_mask']>0,arrays['valid_mask']>0
        if not valid.any() or not lcd.any() or (lcd&~valid).any():raise ValueError('Invalid visible LCD target')
        if 'protected_mask' in arrays and ((arrays['protected_mask']>0)&(~valid|lcd)).any():
            raise ValueError('Protected analog target overlaps LCD or invalid sensor pixels')
        pixels[split].add(hashlib.sha256(arrays['image'].tobytes()).hexdigest())
        rows[split].append(row)
        if progress and index%250==0:print(f'LCD images/labels {index}/{len(annotations)}',flush=True)
    if any(len(splits)>1 for splits in source_splits.values()):raise ValueError('Source family crosses splits')
    if any(not value for value in rows.values()):raise ValueError('Train, validation and held-out test are required')
    for a,b in (('train','val'),('train','test'),('val','test')):
        if pixels[a]&pixels[b]:raise ValueError('Identical decoded images cross splits')
    if recordings['test']&(recordings['train']|recordings['val']):raise ValueError('Held-out test recording overlaps training/validation')
    identity={'manifest_sha256':cp.sha256(manifest_path),'samples_sha256':cp.sha256(root/'samples.jsonl'),
        'sources_sha256':cp.sha256(root/'sources.json'),'protected_files_digest':digest(hashes),
        'protected_file_count':len(hashes),'sample_counts':{k:len(v) for k,v in rows.items()},
        'source_counts':dict(Counter(source['split'] for source in sources)),
        'train_validation_recording_overlap':sorted(recordings['train']&recordings['val']),
        'test_recording_groups':sorted(recordings['test']),'test_recording_disjoint':True,
        'protected_analog_sample_count':sum(bool(row.get('protected_mask')) for row in annotations),
        'split_policy':manifest.get('split_policy'),
        'limitation':'Validation uses temporally separated source families within recordings; test is a separate recording, with annotated masks subject to boundary uncertainty'}
    if progress:print('LCD dataset verified: '+json.dumps(identity['sample_counts']),flush=True)
    return rows,identity


class LCDDataset(Dataset):
    def __init__(self,root,rows):self.root=Path(root).resolve();self.rows=list(rows)
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index]
        with Image.open(self.root/row['image']) as image:gray=np.asarray(image).copy()
        def mask(name):
            if row.get(name) is None:return torch.zeros(1,512,512)
            with Image.open(self.root/row[name]) as image:array=np.asarray(image).copy()>0
            return torch.from_numpy(array).float().unsqueeze(0)
        return {'image':normalize_gray(gray),'lcd':mask('lcd_mask'),'valid':mask('valid_mask'),
                'protected':mask('protected_mask'),'protected_available':bool(row.get('protected_mask')),'index':index}


class EpochShuffleSampler(Sampler):
    def __init__(self,rows,seed):self.rows=list(rows);self.seed=seed;self.epoch=0;self.ids_digest=digest([r['id'] for r in rows])
    def __len__(self):return len(self.rows)
    def __iter__(self):return iter(torch.randperm(len(self.rows),generator=torch.Generator().manual_seed(self.seed+7919*self.epoch)).tolist())
    def set_epoch(self,epoch):self.epoch=int(epoch)
    def state_dict(self):return {'epoch':self.epoch,'seed':self.seed,'ids_digest':self.ids_digest,'policy':'each sample once; deterministic shuffled epoch'}
    def load_state_dict(self,state):
        if any(state.get(k)!=v for k,v in self.state_dict().items() if k!='epoch'):raise ValueError('LCD sampler identity mismatch')
        self.set_epoch(state['epoch'])


def load_model(checkpoint,device='cuda'):
    device=require_colab_cuda(str(device))
    payload,_=cp.load_export(checkpoint)
    if payload.get('architecture')!=ARCHITECTURE_ID or payload.get('preprocessing')!=PREPROCESSING_ID:
        raise ValueError('Incompatible LCD architecture/preprocessing')
    model=LCDSegmenter(weights=None).to(device)
    model.load_state_dict(payload['model_state_dict'],strict=True);model.eval()
    return model,payload


def _identity(config,data,sampler,device):
    settings=asdict(config)
    for key in ('data_dir','output_dir','epochs','device'):settings.pop(key)
    runtime={'torch':str(torch.__version__),'torchvision':str(torchvision.__version__),'numpy':str(np.__version__),
             'opencv':str(cv2.__version__),'pillow':str(PIL.__version__),'device':device.type,'device_name':torch.cuda.get_device_name(device)}
    return {'architecture':ARCHITECTURE_ID,'preprocessing':PREPROCESSING_ID,'dataset':data,
        'training_config':settings,'runtime':runtime,'opencv_version':str(cv2.__version__),
        'numpy_version':str(np.__version__),'sampler':{k:v for k,v in sampler.state_dict().items() if k!='epoch'},
        'implementation_hashes':{name:cp.sha256(ROOT/name) for name in ('learned_lcd_v1/model.py',
            'learned_lcd_v1/train.py','learned_lcd_v1/checkpoints.py','learned_lcd_v1/evaluate.py',
            'learned_lcd_v1/inference.py','real_adaptation_v3/train_needle.py')},
        'resume_policy':'same dataset/code/runtime/settings; total epoch limit may increase',
        'scheduler_policy':'constant per-group learning rates; explicit null scheduler state'}


def train(config,mode='imagenet',checkpoint=None,*,stop_after_epochs=None):
    if isinstance(config,dict):config=TrainConfig(**config)
    device=require_colab_cuda(config.device)
    cv2.setNumThreads(1)
    if mode not in ('imagenet','fresh','resume_best','resume_latest','resume_explicit'):raise ValueError('Invalid mode')
    if min(config.epochs,config.batch_size,config.patience,config.progress_every_batches)<1 or config.num_workers!=0 or config.warmup_epochs<0:
        raise ValueError('Invalid training counts; exact resume requires num_workers=0')
    if any(not math.isfinite(v) or v<=0 for v in (config.decoder_lr,config.encoder_lr,config.grad_clip)) or config.analog_weight<0:
        raise ValueError('Invalid optimization settings')
    rows,data_identity=verify_dataset(config.data_dir)
    sampler=EpochShuffleSampler(rows['train'],config.seed)
    identity=_identity(config,data_identity,sampler,device)
    selected,recovery=cp.resolve(config.output_dir,mode,identity,checkpoint)
    source=cp.load(selected['path'],identity)[0] if selected else None
    if source and source['epoch']>=config.epochs:raise ValueError('Selected checkpoint reached the epoch limit; increase epochs explicitly to continue')
    cp.seed(config.seed)
    print('Constructing LCD segmentation model ('+mode+')...',flush=True)
    model=LCDSegmenter(weights='imagenet' if mode=='imagenet' else None).to(device)
    initialization=pretrained_identity() if mode=='imagenet' else {'mode':'fresh','encoder_initialization':'random','decoder_initialization':'random','optimizer_restored':False}
    optimizer=torch.optim.AdamW([{'params':list(model.features.parameters()),'lr':config.encoder_lr,'name':'encoder'},
        {'params':[p for name,p in model.named_parameters() if not name.startswith('features.')],'lr':config.decoder_lr,'name':'decoder'}],weight_decay=config.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=config.amp)
    generators={name:torch.Generator().manual_seed(config.seed+offset) for name,offset in (('train',1),('val',2))}
    train_loader=DataLoader(LCDDataset(config.data_dir,rows['train']),batch_size=config.batch_size,sampler=sampler,
        generator=generators['train'],num_workers=0,pin_memory=True)
    val_loader=DataLoader(LCDDataset(config.data_dir,rows['val']),batch_size=config.batch_size,shuffle=False,
        generator=generators['val'],num_workers=0,pin_memory=True)
    epoch=0;history=[];best_epoch=0;best_score=float('inf');bad_epochs=0;best_reference=None;latest=None
    if source:
        model.load_state_dict(source['model_state_dict'],strict=True);optimizer.load_state_dict(source['optimizer_state_dict'])
        scaler.load_state_dict(source['scaler_state_dict']);sampler.load_state_dict(source['sampler_state'])
        epoch=source['epoch'];history=deepcopy(source['history']);best_epoch=source['best_epoch'];best_score=source['best_score'];bad_epochs=source['bad_epochs']
        initialization=source['initialization']
        for name,generator in generators.items():generator.set_state(source['loader_generator_states'][name])
        cp.restore_rng(source['rng_state'])
        best_reference=({'path':selected['path'],'sha256':selected['sha256'],'epoch':best_epoch} if best_epoch==epoch else source['best_checkpoint'])
        if not best_reference or cp.sha256(best_reference['path'])!=best_reference['sha256']:raise ValueError('Earlier best full checkpoint unavailable')
        earlier,_=cp.load(best_reference['path'],identity)
        if earlier['epoch']!=best_epoch or earlier['best_score']!=best_score:raise ValueError('Best full checkpoint differs from resume history')
        latest=selected['path']
    run=Path(config.output_dir)/(datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')+'_'+uuid.uuid4().hex[:8])
    run.mkdir(parents=True,exist_ok=False)
    cp.atomic_json(run/'identity.json',identity);cp.atomic_json(run/'dataset_integrity.json',data_identity)
    def save_epoch():
        nonlocal latest,best_reference
        path=run/f'epoch_{epoch:05d}_{uuid.uuid4().hex[:10]}.pt'
        payload={'format_version':cp.FORMAT_VERSION,'checkpoint_kind':cp.KIND,'identity':identity,'config':asdict(config),
            'model_state_dict':cp.cpu(model.state_dict()),'optimizer_state_dict':cp.cpu(optimizer.state_dict()),
            'scaler_state_dict':scaler.state_dict(),'scheduler_state_dict':None,'rng_state':cp.rng_state(),
            'loader_generator_states':{name:g.get_state() for name,g in generators.items()},'sampler_state':sampler.state_dict(),
            'epoch':epoch,'history':deepcopy(history),'best_epoch':best_epoch,'best_score':best_score,'bad_epochs':bad_epochs,
            'best_checkpoint':None if best_epoch==epoch else best_reference,'initialization':initialization}
        marker=cp.save(payload,path);latest=str(path)
        if best_epoch==epoch:best_reference={'path':str(path),'sha256':marker['sha256'],'epoch':epoch}
        cp.atomic_json(run/'latest_checkpoint.json',marker);cp.atomic_json(run/'best_checkpoint.json',best_reference)
        cp.atomic_json(run/'history.json',history)
        print(f'Saved verified full LCD epoch {epoch}; best epoch {best_epoch}',flush=True)
    if not source:
        print('LCD epoch 0 validation baseline...',flush=True)
        metrics,_=evaluate_model(model,val_loader,device,analog_weight=config.analog_weight,progress_every=config.progress_every_batches)
        best_score=metrics['selection_score'];history=[{'epoch':0,'stage':'initialization','validation':metrics}]
        save_epoch()
    completed=0;stop_reason=None
    for epoch_next in range(epoch+1,config.epochs+1):
        if stop_after_epochs is not None and completed>=stop_after_epochs:stop_reason='requested_checkpoint_boundary_stop';break
        sampler.set_epoch(epoch_next)
        stage=configure_stage(model,epoch_next,config.warmup_epochs,fresh=initialization['mode']=='fresh')
        total=0.;count=0;start=time.monotonic()
        for batch_index,batch in enumerate(train_loader,1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',enabled=config.amp):
                logits=model(batch['image'].to(device,non_blocking=True))['lcd_logits']
                loss,_=segmentation_loss(logits,batch['lcd'].to(device),batch['valid'].to(device),
                    batch['protected'].to(device),batch['protected_available'].to(device),analog_weight=config.analog_weight)
            if not torch.isfinite(loss):raise ValueError('Nonfinite LCD loss; earlier Drive checkpoints remain valid')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.grad_clip,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update()
            total+=float(loss.detach())*len(batch['index']);count+=len(batch['index'])
            if batch_index==1 or batch_index%config.progress_every_batches==0:
                print(f'LCD epoch {epoch_next}/{config.epochs} batch {batch_index}/{len(train_loader)} loss={total/count:.5f} elapsed={time.monotonic()-start:.0f}s',flush=True)
        metrics,_=evaluate_model(model,val_loader,device,analog_weight=config.analog_weight,progress_every=config.progress_every_batches)
        epoch=epoch_next;history.append({'epoch':epoch,'stage':stage,'train_loss':total/count,'training_images':count,'validation':metrics})
        score=metrics['selection_score'];meaningful=score<best_score-config.min_delta
        if score<best_score:best_epoch=epoch;best_score=score
        bad_epochs=0 if meaningful else bad_epochs+1
        save_epoch();completed+=1
        print(json.dumps({'epoch':epoch,'val_dice':metrics['raw']['macro_dice'],'val_iou':metrics['raw']['macro_iou'],
            'gate_ready':metrics['gate_selection']['ready_for_integration'],'coverage':metrics['coverage_fraction'],'best_epoch':best_epoch}),flush=True)
        if bad_epochs>=config.patience and epoch>config.warmup_epochs:stop_reason='early_stopping';break
    if stop_reason is None:stop_reason='target_reached'
    best,_=cp.load(best_reference['path'],identity)
    model.load_state_dict(best['model_state_dict'],strict=True)
    selection=history[best_epoch]['validation']['gate_selection']
    gate=selection['gate_config'];ready=selection['ready_for_integration']
    validation_metrics,_=evaluate_model(model,val_loader,device,analog_weight=config.analog_weight,
        fixed_gate_config=gate,output_dir=run/'validation',progress_every=config.progress_every_batches)
    export=run/'best_model.pt'
    cp.durable._durable_torch_save({'format_version':cp.FORMAT_VERSION,'checkpoint_kind':cp.INFERENCE_KIND,
        'architecture':ARCHITECTURE_ID,'preprocessing':PREPROCESSING_ID,'model_state_dict':cp.cpu(model.state_dict()),
        'best_epoch':best_epoch,'identity':identity,'initialization':initialization,'gate_config':gate,
        'ready_for_integration':ready,'gate_selection':selection,'validation_metrics':validation_metrics,
        'contains_optimizer_state':False},export)
    test_metrics=None;test_status='not_evaluated_before_training_complete'
    ledger=Path(config.output_dir)/('heldout_test_'+data_identity['manifest_sha256'][:16]+'.json')
    if stop_reason in ('early_stopping','target_reached'):
        if ledger.exists():test_status='already_evaluated_or_started_for_this_dataset; no automatic repeat'
        else:
            cp.atomic_json(ledger,{'status':'started','dataset_manifest_sha256':data_identity['manifest_sha256'],
                'checkpoint_sha256':cp.sha256(export),'checkpoint':str(export),'output':str(run/'test')})
            test_loader=DataLoader(LCDDataset(config.data_dir,rows['test']),batch_size=config.batch_size,shuffle=False,num_workers=0)
            test_metrics,_=evaluate_model(model,test_loader,device,analog_weight=config.analog_weight,
                fixed_gate_config=gate,output_dir=run/'test',progress_every=config.progress_every_batches)
            test_status='evaluated_once_after_model_and_gate_selection'
            cp.atomic_json(ledger,{'status':'complete','dataset_manifest_sha256':data_identity['manifest_sha256'],
                'checkpoint_sha256':cp.sha256(export),'checkpoint':str(export),'metrics_path':str(run/'test/metrics.json'),
                'model_or_threshold_selection_used_test':False})
    report={'run_dir':str(run),'best_model_path':str(export),'best_epoch':best_epoch,'completed_epoch':epoch,
        'epochs_completed_this_call':completed,'stop_reason':stop_reason,'training_complete':stop_reason in ('early_stopping','target_reached'),
        'latest_checkpoint_path':latest,'best_checkpoint_path':best_reference['path'],'ready_for_integration':ready,
        'gate_config':gate,'gate_selection':selection,'validation_metrics':validation_metrics,'test_metrics':test_metrics,
        'test_evaluation_status':test_status,'test_ledger':str(ledger),'identity':identity,'initialization':initialization,
        'resume':{'mode':mode,'source_checkpoint':selected['path'] if selected else None,'full_state_restored':source is not None,'recovery_diagnostics':recovery},
        'metric_scope':'LCD mask agreement only. Model scores are not calibrated confidence; analog protection is unmeasured when annotations are absent.'}
    cp.atomic_json(run/'training_report.json',report)
    cp.atomic_json(Path(config.output_dir)/'latest_run.json',{'run_dir':str(run),'report_path':str(run/'training_report.json')})
    print('LCD training report: '+str(run/'training_report.json'),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--mode',choices=['imagenet','fresh','resume_best','resume_latest','resume_explicit'],default='imagenet')
    parser.add_argument('--checkpoint');args=parser.parse_args()
    print(json.dumps(train(json.loads(Path(args.config).read_text(encoding='utf-8')),mode=args.mode,checkpoint=args.checkpoint),indent=2))


if __name__=='__main__':main()
