"""Deterministic full-circle camera-roll variants of the approved 500 images.

Positive angles turn the whole dashboard clockwise. Speed never changes.
No generative image service is used. Images and geometric labels share one affine.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import shutil
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ANGLES = [0, 30, 60, 90, 120, 150, 180, -150, -120, -90, -60, -30]
UNSEEN_ANGLES = [15, 45, 75, 105, 135, 165, -165, -135, -105, -75, -45, -15]
SIZE = 512

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def rotation_matrix(angle_degrees, size=512):
    """3x3 source-to-output transform, clockwise in x-right/y-down coordinates."""
    result = np.eye(3)
    result[:2] = cv2.getRotationMatrix2D(((size-1)/2, (size-1)/2), -float(angle_degrees), 1.0)
    return result

def transform_points(points, matrix):
    points = np.asarray(points, dtype=float)
    xyw = np.c_[points, np.ones(len(points))] @ np.asarray(matrix).T
    return xyw[:, :2] / xyw[:, 2:]

def rotate_sample(parent, image, mask, angle_degrees, sample_id=None):
    """Return {image, mask, label}; label preserves original pose group and split."""
    if image.shape != (SIZE, SIZE) or mask.shape != image.shape:
        raise ValueError('Expected 512 x 512 image and mask')
    label = copy.deepcopy(parent)
    R = rotation_matrix(angle_degrees)
    sid = sample_id or f"{parent['sample_id']}_r{int(angle_degrees)%360:03d}"
    # Constant fill cannot invent mirrored numbers/needles. No zoom or rescaling.
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    fill = int(np.median(border))
    if angle_degrees == 0:
        rotated, rotated_mask = image.copy(), mask.copy()
    else:
        rotated = cv2.warpAffine(image, R[:2], (SIZE, SIZE), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
        rotated_mask = cv2.warpAffine(mask, R[:2], (SIZE, SIZE), flags=cv2.INTER_NEAREST,
                                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    names = list(parent['needle_position_px'])
    points = transform_points([parent['needle_position_px'][n] for n in names], R)
    if not np.all((points >= 0) & (points <= SIZE-1)):
        raise ValueError(f'Keypoint clipped: {sid}')
    label['needle_position_px'] = dict(zip(names, points.tolist()))
    label['needle_position_normalized'] = dict(zip(names, (points/(SIZE-1)).tolist()))
    v = np.array(label['needle_position_px']['tip']) - label['needle_position_px']['pivot']
    label['needle_angle_image_degrees_clockwise_from_right'] = float(np.degrees(np.arctan2(v[1], v[0])) % 360)
    label['source_to_output_homography_3x3'] = (R @ np.array(parent['source_to_output_homography_3x3'])).tolist()
    label.update(sample_id=sid, image=f'images/{sid}.png', needle_mask=f'masks/{sid}.png',
                 parent_sample_id=parent['sample_id'], parent_group_id=parent['group_id'],
                 parent_image_sha256=parent['image_sha256'], rotation_degrees=float(angle_degrees),
                 parent_to_output_affine_3x3=R.tolist(),
                 rotation_convention='clockwise positive, image x right/y down',
                 origin='unchanged_v1_original' if angle_degrees == 0 else 'v1_camera_roll_variant',
                 review_status='automated_geometry_verified_manual_spot_check',
                 keypoints_in_frame=True)
    label.setdefault('augmentation', {})['camera_roll'] = {
        'degrees_clockwise': float(angle_degrees), 'center_xy': [255.5, 255.5],
        'scale': 1.0, 'interpolation': 'linear image; nearest mask',
        'border_mode': 'constant parent-edge median', 'border_value': fill,
        'scope': 'rotates existing crop; missing exterior scene is not reconstructed'}
    return {'image': rotated, 'mask': rotated_mask, 'label': label}

def write_csv(path, rows):
    fields = ['id','sample_id','image','speed_kmh','pivot_x','pivot_y','tip_x','tip_y',
              'pivot_x_normalized','pivot_y_normalized','tip_x_normalized','tip_y_normalized',
              'image_angle_degrees','needle_mask','profile','group_id','dataset_split','origin',
              'anchor_method','inferred_60_influence','review_status','rng_seed',
              'rotation_degrees','parent_sample_id','parent_group_id']
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in rows:
            v = {k: row.get(k) for k in fields}
            for name in ['pivot','tip']:
                v[name+'_x'],v[name+'_y'] = row['needle_position_px'][name]
                v[name+'_x_normalized'],v[name+'_y_normalized'] = row['needle_position_normalized'][name]
            v['image_angle_degrees'] = row['needle_angle_image_degrees_clockwise_from_right']
            v['profile'] = row['augmentation']['photometric']['profile']
            w.writerow(v)

def build(source, output):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing dataset: {output}')
    doc = json.loads((source/'labels.json').read_text())
    parents = doc['samples']
    if Counter(r['dataset_split'] for r in parents) != {'train':400,'validation':100}:
        raise ValueError('Expected original 400/100 split')
    for root in [output, output/'robustness']:
        (root/'images').mkdir(parents=True); (root/'masks').mkdir()
    (output/'source').mkdir()
    source_info = copy.deepcopy(doc['source'])
    shutil.copyfile(source/source_info['file'], output/source_info['file'])
    shutil.copyfile(source/'calibration.json', output/'calibration.json')
    tasks = [(p, a, False) for p in parents for a in ANGLES]
    tasks += [(p, a, True) for p in parents if p['dataset_split']=='validation' for a in UNSEEN_ANGLES]
    started = time.time()
    def process(task):
        p,a,unseen = task
        root = output/'robustness' if unseen else output
        image_path = source/p['image']
        if sha256(image_path) != p['image_sha256']:
            raise ValueError('Parent image hash changed')
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(source/p['needle_mask']), cv2.IMREAD_GRAYSCALE)
        result = rotate_sample(p,image,mask,a)
        row = result['label']
        for key,array in [('image',result['image']),('needle_mask',result['mask'])]:
            dest = root/row[key]
            if a==0:
                shutil.copyfile(source/p[key],dest)
            elif not cv2.imwrite(str(dest),array,[cv2.IMWRITE_PNG_COMPRESSION,6]):
                raise IOError(str(dest))
            row[key+'_sha256'] = sha256(dest)
        row['evaluation_set'] = 'unseen_angles_robustness' if unseen else 'main'
        return unseen,row
    rows,robust = [],[]
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(unseen,row) in enumerate(pool.map(process,tasks),1):
            (robust if unseen else rows).append(row)
            if i%600==0: print(f'Generated {i}/{len(tasks)} ({time.time()-started:.1f}s)',flush=True)
    for records in [rows,robust]:
        for i,row in enumerate(records,1): row['id']=i
    common = dict(schema_version='3.0',coordinate_convention=doc['coordinate_convention'],
        parent_labels_sha256=sha256(source/'labels.json'),
        rotation_convention='clockwise positive, x right/y down; degrees relative to parent image',
        calibration_status=doc['calibration_status'],
        limitation='One synthetic master. Rotations are correlated variants, not new independent poses or real test data.')
    main = dict(common, count=len(rows), split_counts={'train':4800,'validation':1200},
        source=source_info, angles_degrees=ANGLES, samples=rows,
        split_rule='All descendants retain parent group_id and split. No validation parent is used for training.')
    unseen = dict(common,count=len(robust),split_counts={'validation':1200},samples=robust,
        angles_degrees=UNSEEN_ANGLES,
        evaluation_rule='Before/after robustness only; never early stopping, model selection or training. Same 50 validation poses.')
    for root,document in [(output,main),(output/'robustness',unseen)]:
        write_json(root/'labels.json',document); write_csv(root/'labels.csv',document['samples'])
    for split in ['train','validation']:
        write_csv(output/f'splits/{split}.csv',[r for r in rows if r['dataset_split']==split])
    groups = defaultdict(set)
    for r in rows+robust: groups[r['group_id']].add(r['dataset_split'])
    assert all(len(s)==1 for s in groups.values())
    summary = dict(count=len(rows),new_training_and_validation_variants=5500,retained_originals=500,
        split_counts=main['split_counts'],robustness_count=len(robust),angles_degrees=ANGLES,
        unseen_angles_degrees=UNSEEN_ANGLES,train_pose_groups=len({r['group_id'] for r in rows if r['dataset_split']=='train'}),
        validation_pose_groups=len({r['group_id'] for r in rows if r['dataset_split']=='validation'}),
        pose_group_leakage=False,labels_sha256=sha256(output/'labels.json'),
        robustness_labels_sha256=sha256(output/'robustness/labels.json'),
        source_sha256=source_info['sha256'],size=[512,512],channels=1,
        created_seconds=round(time.time()-started,2),limitation=common['limitation'])
    write_json(output/'dataset_summary.json',summary)
    (output/'README.md').write_text('# Full-circle camera rotation dataset\n\n'
        '6,000 main 512 x 512 grayscale images: 4,800 training + 1,200 validation. '
        'A further 1,200 robustness images use the same validation parents at unseen intermediate angles.\n\n'
        'Positive angles are clockwise. Speed is unchanged; pivot, tip, tick anchor, mask and homography use the exact same affine. '
        'Zero-angle images are byte-identical to v1. Missing corners use constant parent-edge median gray; no reflected ticks or needles. '
        'All variants keep their parent pose group and split. No ring detector or physical calibration is implied.\n\n'
        'Robustness angles must not influence early stopping or model selection. All results remain synthetic development evaluation. '
        'The obscured 60 km/h calibration anchor remains inferred from v1. Real camera testing is still required.\n',encoding='utf-8')
    shutil.copyfile(Path(__file__), output/'generate_rotations.py')
    make_preview(output, rows)
    print(json.dumps(summary,indent=2),flush=True)
    return summary

def make_preview(root, rows):
    selected=[r for r in rows if r['parent_sample_id']=='sample_0205']
    canvas=Image.new('RGB',(4*260,3*300),(22,25,30));draw=ImageDraw.Draw(canvas)
    try:
        font=ImageFont.truetype('DejaVuSans.ttf',17)
        small=ImageFont.truetype('DejaVuSans.ttf',14)
    except OSError:
        font=ImageFont.load_default(); small=ImageFont.load_default()
    for i,r in enumerate(selected):
        x,y=(i%4)*260,(i//4)*300
        im=Image.open(root/r['image']).convert('RGB').resize((256,256))
        # Label overlay is review-only, NEVER model input.
        d=ImageDraw.Draw(im)
        p,t=[np.array(r['needle_position_px'][n])/2 for n in ('pivot','tip')]
        for pt in (p,t): d.ellipse((pt[0]-3,pt[1]-3,pt[0]+3,pt[1]+3),outline=(0,255,180),width=1)
        canvas.paste(im,(x,y));draw.text((x+5,y+260),f"{r['rotation_degrees']:+.0f} deg | {r['speed_kmh']} km/h",font=font,fill='white')
        draw.text((x+5,y+281),'Green marks: label positions',font=small,fill=(145,185,175))
    canvas.save(root/'rotation_preview.jpg',quality=90)

def package(root, archive):
    root,archive=Path(root),Path(archive)
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file(): z.write(p,Path(root.name)/p.relative_to(root))
    digest=sha256(archive)
    archive.with_suffix(archive.suffix+'.sha256').write_text(f'{digest}  {archive.name}\n',encoding='ascii')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'sha256':digest}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--archive',type=Path)
    args=parser.parse_args();build(args.source,args.output)
    if args.archive: package(args.output,args.archive)
