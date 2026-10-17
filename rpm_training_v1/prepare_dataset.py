"""Procedural RPM data on real LCD backgrounds; no ImageGen and no pseudo RPM labels.

Only the renderer's known fill edge supplies positive targets. The synthetic
target is position on a draft visually marked scale, not measured engine RPM.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from preprocess import (BACKGROUND_VALUE, CROP_XYXY, HERE, INPUT_HEIGHT, INPUT_WIDTH,
                        load_calibration, preprocess_aligned, preprocessing_metadata,
                        reference_to_input_matrix, transform_points)

PROJECT = HERE.parent
PROTOTYPE = PROJECT/'rpm_prototype'
SEED = 812107
SOURCES = {
    'train': {'baseline': '07_220558_frame_0000.png', 'appearance': '11_220558_frame_0169.png'},
    'val': {'baseline': '08_220558_frame_0042.png', 'appearance': '13_220558_frame_0254.png'},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def setup_assets():
    assets = HERE/'calibration'
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROTOTYPE/'calibration_draft.json', assets/'calibration_draft.json')
    records = {}
    for split, roles in SOURCES.items():
        records[split] = {}
        for role, name in roles.items():
            src = PROTOTYPE/'review20/aligned'/name
            dst = assets/f'{split}_{role}.png'
            shutil.copy2(src, dst)
            records[split][role] = {'asset': str(dst.relative_to(HERE)).replace('\\','/'),
                                    'original_aligned_file': name, 'sha256': digest(dst),
                                    'uses_numeric_rpm_prediction': False}
    (assets/'preprocessing.json').write_text(json.dumps(preprocessing_metadata(),indent=2),encoding='utf-8')
    (assets/'source_provenance.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    return records


class Renderer:
    def __init__(self, split):
        self.cal = load_calibration()
        self.baseline = cv2.imread(str(HERE/f'calibration/{split}_baseline.png'), cv2.IMREAD_GRAYSCALE)
        appearance = cv2.imread(str(HERE/f'calibration/{split}_appearance.png'), cv2.IMREAD_GRAYSCALE)
        # Observe only the darkening/stripe appearance in a visibly active region.
        # This operation uses no predicted RPM value.
        difference = self.baseline[448:469,198:211].astype(np.float32)-appearance[448:469,198:211].astype(np.float32)
        self.observed_darkening = float(np.clip(np.percentile(difference,80),25,75))
        self.anchors = np.asarray([a['xy'] for a in self.cal['scale_anchors']],np.float32)
        self.rpms = np.asarray([a['rpm'] for a in self.cal['scale_anchors']],np.float32)
        self.left = np.asarray(self.cal['left_edge_xy'],np.float32)
        self.right = np.asarray(self.cal['inner_right_edge_xy'],np.float32)
        self.slope = float(self.cal['nominal_bar_slope_reference'])
        self.supersample = 4
        x0,y0,x1,y1 = CROP_XYXY
        self.xx,self.yy = np.meshgrid(x0+(np.arange((x1-x0)*4)+.5)/4-.5,
                                     y0+(np.arange((y1-y0)*4)+.5)/4-.5)
        lx = np.interp(self.yy,self.left[:,1],self.left[:,0])
        rx = np.interp(self.yy,self.right[:,1],self.right[:,0])
        self.interior = (self.xx>=lx+1.3)&(self.xx<=rx-.9)
        exclusions = np.zeros_like(self.baseline)
        for poly in self.cal['static_exclusion_polygons_xy']:
            cv2.fillPoly(exclusions,[np.int32(poly)],255)
        self.interior &= cv2.resize(exclusions[y0:y1,x0:x1],self.xx.shape[::-1],interpolation=cv2.INTER_NEAREST)==0
        self.high_base = cv2.resize(self.baseline[y0:y1,x0:x1].astype(np.float32),self.xx.shape[::-1],interpolation=cv2.INTER_CUBIC)

    def edge(self, rpm):
        ax = float(np.interp(rpm,self.rpms,self.anchors[:,0]))
        ay = float(np.interp(rpm,self.rpms,self.anchors[:,1]))
        ex = (float(np.interp(ay,self.left[:,1],self.left[:,0]))+float(np.interp(ay,self.right[:,1],self.right[:,0])))/2
        return np.array([ex,ay+self.slope*(ex-ax)],np.float64), (ax,ay)

    def render(self, rpm, params):
        if rpm is None:
            return self.baseline.copy(), None
        edge,(ax,ay) = self.edge(rpm)
        distance = self.yy-(ay+self.slope*(self.xx-ax))
        # The first dark stripe begins exactly at the rendered target edge.
        # Pitch is appearance variation, not a statement about physical RPM/segment.
        pitch,duty = params['stripe_pitch_native_px'],params['stripe_duty']
        phase = np.mod(np.maximum(distance,0),pitch)
        stripe = np.where(phase<pitch*duty,1.0,.10)
        fill = self.interior&(distance>=0)&(self.yy<474)
        delta = self.observed_darkening*params['darkening_multiplier']*stripe
        texture = params['vertical_shading']*(self.yy-400)/100
        rendered = self.high_base - delta*fill + texture*fill
        x0,y0,x1,y1 = CROP_XYXY
        result = self.baseline.copy()
        result[y0:y1,x0:x1] = np.clip(cv2.resize(rendered,(x1-x0,y1-y0),interpolation=cv2.INTER_AREA),0,255).astype(np.uint8)
        return result,edge


def augment(image, mask, point, rng, negative_kind):
    h,w = image.shape
    angle = float(rng.uniform(-4,4)); scale=float(rng.uniform(.95,1.04))
    tx,ty = rng.uniform(-4,4,2); shear=float(rng.uniform(-.025,.025))
    affine = np.vstack([cv2.getRotationMatrix2D((w/2,h/2),angle,scale),[0,0,1]])
    affine[0,1]+=shear; affine[0,2]+=tx; affine[1,2]+=ty
    corners = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
    offsets = rng.uniform(-1.8,1.8,(4,2)).astype(np.float32)
    perspective = cv2.getPerspectiveTransform(corners,corners+offsets)
    matrix=perspective@affine
    output=cv2.warpPerspective(image,matrix,(w,h),flags=cv2.INTER_LINEAR,borderValue=BACKGROUND_VALUE)
    warped_mask=cv2.warpPerspective(mask,matrix,(w,h),flags=cv2.INTER_NEAREST,borderValue=0)
    transformed=transform_points([point],matrix)[0] if point is not None else None
    gain=float(rng.uniform(.84,1.10)); bias=float(rng.uniform(-15,12)); gamma=float(rng.uniform(.88,1.13))
    gradx,grady=rng.uniform(-18,18,2)
    yy,xx=np.mgrid[:h,:w]
    values=255*np.power(output.astype(np.float32)/255,gamma)*gain+bias+gradx*(xx/w-.5)+grady*(yy/h-.5)
    noise=float(rng.uniform(.3,2.4))
    values+=rng.normal(0,noise,values.shape)
    sigma=float(rng.choice([0,.25,.45,.65,.85]))
    if sigma>0: values=cv2.GaussianBlur(values,(0,0),sigma)
    negative_params={}
    if negative_kind=='occluded_edge':
        cy=float(transformed[1]) if transformed is not None else float(h*.6)
        y0=max(0,int(cy-rng.uniform(28,44))); y1=min(h,int(cy+rng.uniform(28,44)))
        shade=int(rng.integers(65,240));values[y0:y1,:]=shade
        negative_params={'occlusion_xyxy':[0,y0,w,y1],'occlusion_value':shade}
    elif negative_kind=='severe_blur':
        severe_sigma=float(rng.uniform(10,15))
        values=cv2.GaussianBlur(values,(0,0),severe_sigma)
        # Severe LCD contrast loss makes the endpoint intentionally unresolved.
        values=values.mean()+(values-values.mean())*.08
        negative_params={'severe_sigma_input_px':severe_sigma,'remaining_contrast_fraction':.08}
    elif negative_kind=='washout':
        values=248+rng.normal(0,.6,values.shape)
        negative_params={'washout_mean':248}
    output=np.clip(values,0,255).astype(np.uint8)
    quality=int(rng.integers(58,96))
    ok,encoded=cv2.imencode('.jpg',output,[cv2.IMWRITE_JPEG_QUALITY,quality])
    if not ok: raise RuntimeError('JPEG augmentation failed')
    output=cv2.imdecode(encoded,cv2.IMREAD_GRAYSCALE)
    output[warped_mask==0]=BACKGROUND_VALUE
    return output,transformed,{'seed':None,'rotation_deg':angle,'scale':scale,
        'translation_xy_input_px':[float(tx),float(ty)],'shear':shear,
        'perspective_corner_offsets_input_px':offsets.tolist(),'matrix_input_to_augmented':matrix.tolist(),
        'brightness_gain':gain,'brightness_bias':bias,'gamma':gamma,
        'gradient_xy':[float(gradx),float(grady)],'noise_sigma':noise,'blur_sigma_input_px':sigma,
        'jpeg_quality':quality,'negative_effect':negative_params}


def make_contact(rows, root, destination, title, max_items=40):
    cols=8; count=min(max_items,len(rows)); thumb_w,thumb_h=144,192
    tile_h=234
    canvas=Image.new('RGB',(cols*thumb_w,64+((count+cols-1)//cols)*tile_h),(19,28,41))
    draw=ImageDraw.Draw(canvas)
    fpath=Path(r'C:\Windows\Fonts\segoeui.ttf')
    font=ImageFont.truetype(str(fpath),14) if fpath.exists() else ImageFont.load_default()
    draw.text((14,12),title,fill='white',font=font)
    draw.text((14,34),'Green dot: synthetic edge target. Values use draft scale, not verified engine RPM.',fill=(182,195,210),font=font)
    for i,row in enumerate(rows[:count]):
        x=(i%cols)*thumb_w;y=64+(i//cols)*tile_h
        im=Image.open(root/row['image']).convert('RGB').resize((thumb_w,thumb_h),Image.Resampling.NEAREST)
        d=ImageDraw.Draw(im)
        if row['edge_xy']:
            px,py=row['edge_xy'];px*=thumb_w/INPUT_WIDTH;py*=thumb_h/INPUT_HEIGHT
            d.ellipse((px-3,py-3,px+3,py+3),outline=(0,255,160),width=2)
        canvas.paste(im,(x,y))
        text=(f"~{row['indicated_rpm_approx']:.0f} RPM" if row['readable'] else row['sample_type'].replace('_',' '))
        draw.text((x+4,y+196),text,fill=(110,245,185) if row['readable'] else (250,200,120),font=font)
        draw.text((x+4,y+214),row['id'],fill=(182,195,210),font=font)
    canvas.save(destination,quality=95)


def diagnostics(root, cal):
    folder=root/'real_diagnostics';folder.mkdir(exist_ok=True)
    original=json.loads((PROTOTYPE/'review20/predictions.json').read_text(encoding='utf-8'))['previews']
    rows=[]
    for r in original:
        src=PROTOTYPE/'review20/aligned'/Path(r['original_file']).name
        if not src.exists():continue
        im=cv2.imread(str(src));masked=preprocess_aligned(im,cal)
        target=folder/f"real_{r['preview_index']:02d}.png";cv2.imwrite(str(target),masked)
        rows.append({'id':f"real_{r['preview_index']:02d}",'image':str(target.relative_to(root)).replace('\\','/'),
                     'source_clip':r['clip'],'source_frame':r['source_frame'],
                     'display_state_from_prototype':r['display']['state'],
                     'suppress_rpm':r['display']['suppress_rpm'],
                     'engine_off_message':r['display']['engine_off_message'],
                     'manual_review_status':'pending_user_review',
                     'training_eligible':False,'readable':None,'fill_fraction':None,'edge_xy':None,
                     'indicated_rpm_approx':None,'label_origin':'unlabeled_real_development_diagnostic',
                     'reason':'Same recordings used in development; startup needs external full-LCD gate; no verified RPM targets.'})
    (folder/'labels.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    return len(rows)


def make_real_comparison(root, rows):
    chosen=min((r for r in rows if r['readable']),key=lambda r:abs(r['indicated_rpm_approx']-2600))
    canvas=Image.new('RGB',(840,660),(19,28,41));draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(r'C:\Windows\Fonts\segoeui.ttf',18)
    draw.text((20,12),'Same masked input preprocessing / visual QA',font=font,fill='white')
    real=Image.open(root/'real_diagnostics/real_11.png').convert('RGB').resize((384,512),Image.Resampling.NEAREST)
    synth=Image.open(root/chosen['image']).convert('RGB').resize((384,512),Image.Resampling.NEAREST)
    p=chosen['edge_xy'];d=ImageDraw.Draw(synth);px,py=p[0]*2,p[1]*2
    d.ellipse((px-5,py-5,px+5,py+5),outline=(0,255,160),width=2)
    canvas.paste(real,(20,90));canvas.paste(synth,(436,90))
    draw.text((20,48),'REAL video frame / no verified RPM label',font=font,fill=(182,195,210))
    draw.text((436,48),f"SYNTHETIC / draft scale ~{chosen['indicated_rpm_approx']:.0f}",font=font,fill=(110,245,185))
    draw.text((20,620),'Gear and trip are outside the curved mask. Green dot is renderer-defined edge.',font=font,fill=(182,195,210))
    canvas.save(HERE/'qa/real_vs_synthetic.png')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'dataset')
    parser.add_argument('--preview-only',action='store_true');args=parser.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=True)
    (root/'images/train').mkdir(parents=True,exist_ok=True);(root/'images/val').mkdir(parents=True,exist_ok=True)
    (HERE/'qa').mkdir(exist_ok=True)
    provenance=setup_assets();rows=[];cal=load_calibration()
    split_specs=[('train',320),('val',80)] if not args.preview_only else [('train',12),('val',4)]
    for split,nfamilies in split_specs:
        renderer=Renderer(split)
        for family in range(nfamilies):
            family_seed=SEED+(0 if split=='train' else 100000)+family*101
            rng=np.random.default_rng(family_seed)
            # Whole renderer families are assigned before any augmentation.
            negative=family%5==4
            kinds=['no_resolved_fill','occluded_edge','severe_blur','washout']
            kind=kinds[(family//5)%4] if negative else 'known_synthetic_fill'
            # Low RPM gets extra coverage; full range is explicitly synthetic.
            rpm=float(rng.uniform(500,3500) if family%3==0 else rng.uniform(500,9500))
            if kind=='no_resolved_fill':rpm=None
            params={'stripe_pitch_native_px':float(rng.uniform(3.5,4.05)),
                    'stripe_duty':float(rng.uniform(.55,.72)),
                    'darkening_multiplier':float(rng.uniform(.85,1.18)),
                    'vertical_shading':float(rng.uniform(-1.5,1.5)),
                    'observed_darkening_native_gray':renderer.observed_darkening,
                    'supersample':4,'stripe_phase':'dark stripe starts at synthetic edge'}
            source,edge=renderer.render(rpm,params)
            model,mask=preprocess_aligned(source,cal,return_mask=True)
            p=transform_points([edge],reference_to_input_matrix())[0] if edge is not None else None
            for variant in range(5):
                sample_seed=family_seed+variant+1
                output,point,aug=augment(model,mask,p,np.random.default_rng(sample_seed),kind if negative else None)
                aug['seed']=sample_seed
                ident=f'{split[0]}{family:03d}_{variant}'
                path=f'images/{split}/{ident}.png';cv2.imwrite(str(root/path),output)
                readable=not negative
                if readable and not (0<=point[0]<INPUT_WIDTH and 0<=point[1]<INPUT_HEIGHT):raise AssertionError('Transformed edge left model input')
                rows.append({'id':ident,'image':path,'split':split,'readable':readable,
                    'fill_fraction':float(rpm/12000) if readable else None,
                    'edge_xy':point.tolist() if readable else None,
                    'indicated_rpm_approx':float(rpm) if readable else None,
                    'label_origin':'procedural_known_edge_on_draft_scale' if readable else 'procedural_unresolved_reading',
                    'source_group':f'{split}_render_family_{family:03d}','sample_type':kind,
                    'training_eligible':True,'renderer':{'seed':family_seed,**params,
                        'baseline_asset':provenance[split]['baseline']['asset'],
                        'appearance_asset':provenance[split]['appearance']['asset'],
                        'known_fill_edge_reference_xy':edge.tolist() if readable else None,
                        'numeric_target_from_existing_predictions':False},'augmentation':aug,
                    'calibration_status':'provisional_visual_printed_scale_not_verified_engine_rpm'})
    (root/'labels.jsonl').write_text(''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows),encoding='utf-8')
    shutil.copytree(HERE/'calibration',root/'calibration',dirs_exist_ok=True)
    shutil.copy2(HERE/'preprocess.py',root/'preprocess.py')
    nreal=diagnostics(root,cal)
    for split in ('train','val'):
        split_rows=[r for r in rows if r['split']==split]
        # One variant from consecutive families shows all renderer/negative types.
        selected=[r for r in split_rows if r['id'].endswith('_0')]
        make_contact(selected,root,HERE/f'qa/{split}_contact.jpg',f'{split.upper()} / masked RPM synthetic examples')
    make_real_comparison(root,rows)
    counts={split:{'total':sum(r['split']==split for r in rows),
                  'readable':sum(r['split']==split and r['readable'] for r in rows),
                  'unresolved':sum(r['split']==split and not r['readable'] for r in rows),
                  'families':len({r['source_group'] for r in rows if r['split']==split})} for split in ('train','val')}
    hashes={str(p.relative_to(root)).replace('\\','/'):digest(p) for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('manifest.json',)}
    if len({hashes[r['image']] for r in rows})!=len(rows):raise AssertionError('Duplicate rendered image hash')
    manifest={'schema_version':1,'dataset_id':'triumph_rpm_curved_strip_v1','seed':SEED,'count':len(rows),
        'labels':'labels.jsonl','input':preprocessing_metadata(),'counts':counts,'sha256':hashes,
        'source_provenance':provenance,'real_diagnostic_count':nreal,
        'label_semantics':{'readable':'synthetic edge available vs unresolved; uncalibrated, not startup classifier',
            'fill_fraction':'known rendered draft-scale RPM divided by 12000; null for unresolved',
            'edge_xy':'x,y model input pixel-center coordinates after augmentation; null for unresolved',
            'indicated_rpm_approx':'draft printed-scale interpolation; not verified engine RPM',
            'source_group':'procedural render family assigned before five augmentation variants'},
        'split_limitations':['No procedural family or source-frame asset crosses split.',
            'Both splits share the same physical dashboard, recording session, manually marked calibration and renderer.',
            'Validation measures synthetic generalization only, not real-world RPM accuracy.',
            'Real running examples observed only about 0–3500 RPM; synthetic 500–9500 upper range unvalidated.',
            'Fine physical segment RPM increment remains unknown.',
            'Startup excluded from training; apply full-LCD startup gate before this strip reader.',
            'No-fill is unresolved and never certified 0 RPM.'],
        'tools':'Python/NumPy/OpenCV programmatic rendering; no ImageGen; no existing predicted RPM labels used'}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(root),'counts':counts,'labels_sha256':digest(root/'labels.jsonl'),
                      'manifest_sha256':digest(root/'manifest.json')},indent=2))


if __name__=='__main__':main()
