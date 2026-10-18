"""Segmentation metrics, shared-mask-gate selection, and Colab-only evaluation."""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from learned_lcd_v1.inference import DEFAULT_GATE_CONFIG, gate_lcd_probability, validate_gate_config
from learned_lcd_v1.model import segmentation_loss, require_colab_cuda

THRESHOLDS=(.50,.60,.70,.80,.90)
MIN_VALIDATION_PRECISION=.98
MIN_VALIDATION_ACCEPTED_RECALL=.90
MIN_VALIDATION_POSITIVE_COVERAGE=.80
BOUNDARY_TOLERANCE_PX=2


def mask_metrics(prediction,target,valid,protected=None,*,boundary=True):
    p,t,v=(np.asarray(a)>0 for a in (prediction,target,valid))
    if p.shape!=t.shape or p.shape!=v.shape or p.ndim!=2 or not v.any():
        raise ValueError('Expected matching binary masks with valid sensor pixels')
    p,t=p&v,t&v
    tp,fp,fn=int((p&t).sum()),int((p&~t).sum()),int((~p&t).sum())
    result={'target_pixels':int(t.sum()),'predicted_pixels':int(p.sum()),'true_positive_pixels':tp,
            'false_positive_pixels':fp,'false_negative_pixels':fn,
            'iou':tp/(tp+fp+fn) if tp+fp+fn else None,
            'dice':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'precision':tp/(tp+fp) if tp+fp else None,
            'recall':tp/(tp+fn) if tp+fn else None,
            'lcd_pixels_left_exposed_fraction':fn/(tp+fn) if tp+fn else None,
            'background_false_mask_fraction':fp/int((v&~t).sum()) if (v&~t).any() else None,
            'protected_analog_false_mask_fraction':None,'protected_analog_false_mask_pixels':None,
            'boundary_f1':None,'boundary_p95_error_px':None,'boundary_tolerance_px':BOUNDARY_TOLERANCE_PX,
            'boundary_excludes_sensor_cut':True}
    if protected is not None:
        protected=(np.asarray(protected)>0)&v&~t
        if protected.any():
            hit=int((p&protected).sum())
            result.update(protected_analog_false_mask_fraction=hit/int(protected.sum()),protected_analog_false_mask_pixels=hit)
    if boundary:
        # Ignore annotation boundaries created by clipping to the sensor footprint.
        inner=cv2.erode(v.astype(np.uint8),np.ones((7,7),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
        def edge(mask):
            eroded=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
            return mask&~eroded&inner
        pb,tb=edge(p),edge(t)
        result.update(predicted_boundary_pixels=int(pb.sum()),target_boundary_pixels=int(tb.sum()))
        if tb.any() and pb.any():
            dt=cv2.distanceTransform((~tb).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            dp=cv2.distanceTransform((~pb).astype(np.uint8),cv2.DIST_L2,cv2.DIST_MASK_PRECISE)
            distances_p,distances_t=dt[pb],dp[tb]
            precision=float(np.mean(distances_p<=BOUNDARY_TOLERANCE_PX))
            recall=float(np.mean(distances_t<=BOUNDARY_TOLERANCE_PX))
            result['boundary_f1']=2*precision*recall/(precision+recall) if precision+recall else 0.
            result['boundary_p95_error_px']=float(np.percentile(np.r_[distances_p,distances_t],95))
        elif tb.any():
            result['boundary_f1']=0.
    return result


def summarize_masks(rows,field='raw'):
    metrics=[row[field] for row in rows]
    positive=[m for m in metrics if m['target_pixels']>0]
    def mean(name,records=positive,missing=None):
        values=[r[name] if r.get(name) is not None else missing for r in records]
        values=[v for v in values if v is not None]
        return float(np.mean(values)) if values else None
    return {'count':len(rows),'positive_count':len(positive),'absent_lcd_count':len(metrics)-len(positive),
        'macro_iou':mean('iou',missing=0.),'macro_dice':mean('dice',missing=0.),
        'macro_precision_all_positives':mean('precision',missing=0.),'macro_recall_all_positives':mean('recall',missing=0.),
        'macro_boundary_f1':mean('boundary_f1'),'mean_boundary_p95_error_px':mean('boundary_p95_error_px'),
        'boundary_measurable_count':sum(m.get('boundary_p95_error_px') is not None for m in metrics),
        'macro_lcd_pixels_left_exposed_fraction':mean('lcd_pixels_left_exposed_fraction'),
        'macro_background_false_mask_fraction':mean('background_false_mask_fraction',metrics),
        'macro_protected_analog_false_mask_fraction':mean('protected_analog_false_mask_fraction',metrics),
        'protected_analog_label_count':sum(m.get('protected_analog_false_mask_fraction') is not None for m in metrics),
        'protected_analog_intrusion_image_count':sum((m.get('protected_analog_false_mask_pixels') or 0)>0 for m in metrics),
        'absent_lcd_false_mask_image_count':sum(m['target_pixels']==0 and m['predicted_pixels']>0 for m in metrics)}


def select_gate(rows):
    """Predeclared validation rule; never consult the test or video outputs."""
    candidates=[]
    positives=sum(row['raw']['target_pixels']>0 for row in rows)
    if not positives: raise ValueError('LCD gate selection requires positive validation examples')
    for threshold in THRESHOLDS:
        key=f'{threshold:.2f}'
        selected=[row['candidates'][key] for row in rows]
        accepted=[r for r in selected if r['gate']['accepted'] and r['metrics']['target_pixels']>0]
        precision=float(np.mean([r['metrics']['precision'] or 0. for r in accepted])) if accepted else 0.
        accepted_recall=float(np.mean([r['metrics']['recall'] or 0. for r in accepted])) if accepted else 0.
        coverage=len(accepted)/positives
        all_recall=sum((r['metrics']['recall'] or 0.) for r in selected if r['metrics']['target_pixels']>0)/positives
        candidates.append({'pixel_threshold':threshold,'accepted_positive_count':len(accepted),
            'positive_count':positives,'accepted_positive_coverage':coverage,'accepted_macro_precision':precision,
            'accepted_macro_recall':accepted_recall,
            'all_positive_macro_recall_rejections_zero':all_recall,
            'rejection_reasons':dict(Counter(r['gate']['reason'] for r in selected if not r['gate']['accepted'])),
            'eligible':precision>=MIN_VALIDATION_PRECISION and accepted_recall>=MIN_VALIDATION_ACCEPTED_RECALL and coverage>=MIN_VALIDATION_POSITIVE_COVERAGE})
    eligible=[r for r in candidates if r['eligible']]
    # An ineligible configuration is retained only for transparent diagnostics.
    pool=eligible or candidates
    best=max(pool,key=lambda r:(r['all_positive_macro_recall_rejections_zero'],r['accepted_macro_precision'],r['pixel_threshold']))
    return {'ready_for_integration':bool(eligible),'gate_config':{**DEFAULT_GATE_CONFIG,'pixel_threshold':best['pixel_threshold']},
        'selected_validation_result':best,'candidates':candidates,
        'policy':{'minimum_accepted_macro_precision':MIN_VALIDATION_PRECISION,
                  'minimum_accepted_macro_recall':MIN_VALIDATION_ACCEPTED_RECALL,
                  'minimum_accepted_positive_coverage':MIN_VALIDATION_POSITIVE_COVERAGE,
                  'objective':'maximize all-positive recall, treating rejections as zero',
                  'thresholds':list(THRESHOLDS),'other_gate_parameters_tuned':False,
                  'scores_are_calibrated_confidence':False,'selected_on':'validation_only'}}


def summarize_evaluation(rows,gate_selection,combined_loss,analog_weight):
    raw=summarize_masks(rows)
    if raw['positive_count']==0: raise ValueError('LCD evaluation requires labeled positive examples')
    selected_key=f'{gate_selection["gate_config"]["pixel_threshold"]:.2f}'
    selected_rows=[{**row,'gated':row['candidates'][selected_key]['metrics']} for row in rows]
    accepted=sum(row['candidates'][selected_key]['gate']['accepted'] for row in rows)
    segmentation_score=1-raw['macro_dice']+analog_weight*(raw['macro_protected_analog_false_mask_fraction'] or 0.)
    selectable='ready_for_integration' in gate_selection
    result={'raw_threshold':.5,'raw':raw,'gated':summarize_masks(selected_rows,'gated'),
        'accepted_count':accepted,'rejected_count':len(rows)-accepted,'coverage_fraction':accepted/len(rows),
        'rejection_reasons':dict(Counter(row['candidates'][selected_key]['gate']['reason'] for row in rows if not row['candidates'][selected_key]['gate']['accepted'])),
        'segmentation_score':segmentation_score,
        'selection_score':segmentation_score+(0 if gate_selection.get('ready_for_integration') else 2+analog_weight) if selectable else None,
        'selection_definition':'eligible validation gate first; then one minus mean Dice on ALL positive images at0.5 plus analog_weight times protected false-mask fraction; no test selection score',
        'combined_loss':combined_loss,'gate_selection':gate_selection,'groups':{},
        'scope':'LCD segmentation agreement with reviewed masks; not speed accuracy or calibrated correctness'}
    for name in ('source_group','rotation_bucket','appearance_group','augmentation_group'):
        groups={}
        for value in sorted({row[name] for row in rows}):
            subset=[row for row in selected_rows if row[name]==value]
            accepted_group=sum(row['candidates'][selected_key]['gate']['accepted'] for row in subset)
            groups[value]={'raw':summarize_masks(subset),'gated':summarize_masks(subset,'gated'),
                           'coverage_fraction':accepted_group/len(subset)}
        result['groups'][name]=groups
    return result


@torch.no_grad()
def evaluate_model(model,loader,device,*,analog_weight=.5,progress_every=50,fixed_gate_config=None,output_dir=None):
    require_colab_cuda(str(device))
    cv2.setNumThreads(1)
    model.eval();rows=[];total=0.;count=0
    configs=([{**DEFAULT_GATE_CONFIG,'pixel_threshold':value} for value in THRESHOLDS]
             if fixed_gate_config is None else [validate_gate_config(fixed_gate_config)])
    output=Path(output_dir) if output_dir else None
    if output: (output/'masks').mkdir(parents=True,exist_ok=True)
    for number,batch in enumerate(loader,1):
        image=batch['image'].to(device,non_blocking=True)
        logits=model(image)['lcd_logits']
        loss,_=segmentation_loss(logits,batch['lcd'].to(device),batch['valid'].to(device),
              batch['protected'].to(device),batch['protected_available'].to(device),analog_weight=analog_weight)
        if not torch.isfinite(loss): raise ValueError('Nonfinite LCD evaluation loss')
        probabilities=logits.float().sigmoid().squeeze(1).cpu().numpy()
        total+=float(loss)*len(probabilities);count+=len(probabilities)
        for i,probability in enumerate(probabilities):
            record=loader.dataset.rows[int(batch['index'][i])]
            target=batch['lcd'][i,0].numpy()>0;valid=batch['valid'][i,0].numpy()>0
            protected=batch['protected'][i,0].numpy()>0 if batch['protected_available'][i] else None
            row={'id':record['id'],'source_id':record['source_id'],'source_group':record['source_group'],
                 'rotation_bucket':str(int(float(record.get('rotation_deg',0))%360//30)*30),
                 'appearance_group':str(record.get('lighting',record.get('source_view','unspecified'))),
                 'augmentation_group':'augmented' if record.get('is_augmented',abs(float(record.get('rotation_deg',0)))>1e-8) else 'unaugmented',
                 'raw':mask_metrics(probability>=.5,target,valid,protected),'candidates':{}}
            for config in configs:
                mask,gate=gate_lcd_probability(probability,valid,config)
                predicted=mask if mask is not None else np.zeros_like(target)
                key=f'{config["pixel_threshold"]:.2f}'
                row['candidates'][key]={'gate':gate,'metrics':mask_metrics(predicted,target,valid,protected,boundary=fixed_gate_config is not None)}
                if output and fixed_gate_config is not None:
                    filename=f'{len(rows):06d}.png'
                    Image.fromarray(np.asarray(predicted>0,np.uint8)*255).save(output/'masks'/filename,optimize=True)
                    row['predicted_mask']='masks/'+filename
            rows.append(row)
        if number==1 or number%progress_every==0:
            print(f'LCD validation/evaluation batch {number}/{len(loader)}',flush=True)
    selection=(select_gate(rows) if fixed_gate_config is None else {'gate_config':dict(fixed_gate_config),
        'selected_on':'earlier validation only','fixed_for_this_evaluation':True})
    metrics=summarize_evaluation(rows,selection,total/count,analog_weight)
    if output:
        (output/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        (output/'predictions.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
    return metrics,rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',required=True);parser.add_argument('--dataset',required=True)
    parser.add_argument('--split',choices=['val','test'],default='test');parser.add_argument('--output',required=True)
    parser.add_argument('--batch-size',type=int,default=8);args=parser.parse_args()
    device=require_colab_cuda()
    from learned_lcd_v1.train import load_model,verify_dataset,LCDDataset
    from torch.utils.data import DataLoader
    model,payload=load_model(args.checkpoint,device)
    rows,identity=verify_dataset(args.dataset)
    if identity!=payload['identity']['dataset']: raise ValueError('Evaluation dataset identity differs from training')
    output=Path(args.output)
    if output.exists(): raise FileExistsError('Evaluation output exists; refusing hidden repeated holdout run')
    output.mkdir(parents=True)
    loader=DataLoader(LCDDataset(args.dataset,rows[args.split]),batch_size=args.batch_size,shuffle=False)
    metrics,_=evaluate_model(model,loader,device,fixed_gate_config=payload['gate_config'],output_dir=output)
    print(json.dumps(metrics,indent=2),flush=True)


if __name__=='__main__': main()
