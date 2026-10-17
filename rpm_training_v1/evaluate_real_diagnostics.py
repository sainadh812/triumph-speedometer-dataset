"""Inspect learned predictions on real development frames without inventing truth."""
from pathlib import Path
import json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch

try:
    from .train_rpm import RPMModel, INFERENCE_KIND, ARCHITECTURE_ID,sha256,verify_dataset
except ImportError:
    from train_rpm import RPMModel, INFERENCE_KIND, ARCHITECTURE_ID,sha256,verify_dataset


def evaluate_real(dataset, checkpoint, output):
    dataset,checkpoint,output=map(Path,(dataset,checkpoint,output))
    marker=json.loads(checkpoint.with_suffix('.complete.json').read_text())
    if sha256(checkpoint)!=marker['sha256'] or marker['kind']!=INFERENCE_KIND:
        raise ValueError('Inference weights failed integrity check')
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    _,data_identity=verify_dataset(dataset)
    if payload.get('kind')!=INFERENCE_KIND or payload.get('identity',{}).get('architecture')!=ARCHITECTURE_ID:
        raise ValueError('Unexpected inference model kind or architecture')
    if payload['identity'].get('dataset')!=data_identity:
        raise ValueError('Inference dataset/preprocessing identity does not match the checkpoint')
    model=RPMModel();model.load_state_dict(payload['model_state_dict'],strict=True);model.eval()
    path=dataset/'real_diagnostics/labels.jsonl'
    rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    results=[]
    for row in rows:
        image=Image.open(dataset/row['image']).convert('L')
        tensor=torch.from_numpy(np.asarray(image,dtype=np.float32).copy()/255)[None,None]
        with torch.inference_mode():pred=model(tensor)
        q=float(pred['fill_fraction'][0]);score=float(pred['readability_score'][0]);rpm=q*12000
        suppressed=bool(row.get('suppress_rpm',True))
        reason='external_display_state_gate' if suppressed else 'low_readability_score' if score<.5 else 'outside_synthetic_training_range' if not 500<=rpm<=9500 else 'approximate_prediction'
        results.append({'id':row['id'],'image':row['image'],'source_frame':row['source_frame'],
                        'display_state':row.get('display_state',row.get('display_state_from_prototype')),
                        'readability_score':score,'predicted_fill_fraction':q,
                        'indicated_rpm_approx':rpm if reason=='approximate_prediction' else None,
                        'reason':reason,'ground_truth_rpm':None,'training_eligible':False})
    output.mkdir(parents=True,exist_ok=True)
    (output/'real_development_predictions.json').write_text(json.dumps({'scope':'Same-recording unlabeled development frames; no accuracy computed. State gates are precomputed from the full LCD prototype.',
                                                                     'predictions':results},indent=2))
    font_path=next((p for p in [Path('C:/Windows/Fonts/segoeui.ttf'),Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')] if p.is_file()),None)
    font=ImageFont.truetype(str(font_path),13) if font_path else ImageFont.load_default()
    cols=5;nr=(len(results)+cols-1)//cols
    sheet=Image.new('RGB',(cols*224,64+nr*320),(20,28,40));draw=ImageDraw.Draw(sheet)
    draw.text((16,12),'Real development frames - model predictions, no verified RPM labels',font=font,fill='white')
    for i,row in enumerate(results):
        x=(i%cols)*224;y=64+(i//cols)*320
        sheet.paste(Image.open(dataset/row['image']).convert('RGB'),(x+16,y))
        value=row['indicated_rpm_approx']
        label=f'Approx. {value:.0f} RPM' if value is not None else 'No numeric reading'
        draw.text((x+8,y+260),f"{row['id']} | {label}",font=font,fill='white')
        reason={'external_display_state_gate':'Display state: no reading','low_readability_score':'Low readability score',
                'outside_synthetic_training_range':'Outside training range','approximate_prediction':'Approximate / needs review'}[row['reason']]
        draw.text((x+8,y+281),reason,font=font,fill=(180,196,215))
    sheet.save(output/'real_development_predictions.jpg',quality=94)
    return {'count':len(results),'numeric_predictions':sum(r['indicated_rpm_approx'] is not None for r in results),
            'predictions':str(output/'real_development_predictions.json'),
            'contact_sheet':str(output/'real_development_predictions.jpg'),'accuracy':None}
