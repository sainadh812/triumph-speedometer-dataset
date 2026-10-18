"""Versioned LCD-only horizontal/vertical training mirrors; no model execution."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'dataset'
NEW=ROOT/'mirrored_v2'
BASE_SHA='b5bca7882b2f21671860203e7b16ff44ba0728cc0eaaa8f732bd534e20502835'
SEED=2026101802
NOTE=('Synthetic image reflection, not a physically moved right-side camera: '
      'LCD digits and dial printing are reflected. Parent generating roll is retained; '
      'perspective augmentation and reflection are distinct. No speed/RPM target is changed or created.')
AXES={'horizontal':(1,np.array([[-1.,0,511],[0,1,0],[0,0,1]])),
      'vertical':(0,np.array([[1.,0,0],[0,-1,511],[0,0,1]]))}

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def project(points,h):
    p=np.asarray(points,float);q=np.c_[p,np.ones(len(p))]@np.asarray(h,float).T
    return q[:,:2]/q[:,2:]

def swapped_edges(edges,axis):
    mapping={'left':'right','right':'left'} if axis=='horizontal' else {'top':'bottom','bottom':'top'}
    return [mapping.get(x,x) for x in edges]

def select_parents(rows):
    """600 distinct parents, 200/recording, all 12 bins for every source family."""
    rng=np.random.default_rng(SEED);selected=[]
    groups=sorted({r['source_group'] for r in rows if r['split']=='train'})
    assert len(groups)==3
    for group in groups:
        pool=[r for r in rows if r['split']=='train' and r['is_augmented'] and r['source_group']==group]
        rng.shuffle(pool)
        counts=Counter();bin_counts=Counter();used=set()
        for i in range(200):
            bucket=i%12
            candidates=[r for r in pool if r['id'] not in used and int(r['rotation_deg']//30)==bucket]
            assert candidates
            parent=min(candidates,key=lambda r:(bin_counts[(r['source_id'],bucket)],counts[r['source_id']]))
            selected.append(parent);used.add(parent['id'])
            counts[parent['source_id']]+=1;bin_counts[(parent['source_id'],bucket)]+=1
        assert len(used)==200
        assert all(bin_counts[(s,b)]>=1 for s in counts for b in range(12))
        assert max(counts.values())-min(counts.values())<=2
    assert len({r['id'] for r in selected})==600
    return selected

def load_inputs():
    assert sha(BASE/'manifest.json')==BASE_SHA,'Base manifest changed'
    manifest=json.loads((BASE/'manifest.json').read_text())
    raw=(BASE/'samples.jsonl').read_bytes()
    assert raw.endswith(b'\n')
    rows=[json.loads(line) for line in raw.splitlines() if line.strip()]
    assert len(rows)==4426
    return manifest,raw,rows,select_parents(rows)

def preview(parents):
    qa=NEW/'qa';qa.mkdir(parents=True,exist_ok=True)
    try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    except OSError:font=ImageFont.load_default()
    for group in sorted({p['source_group'] for p in parents}):
        candidates=[p for p in parents if p['source_group']==group]
        chosen=[min(candidates,key=lambda p:abs(p['rotation_deg']-angle)) for angle in (0,90,180,270)]
        panel=Image.new('RGB',(1536,4*540),(24,24,24))
        for row,p in enumerate(chosen):
            raw=cv2.imread(str(BASE/p['image']),0);mask=cv2.imread(str(BASE/p['lcd_mask']),0)
            for col,axis in enumerate(('original','horizontal','vertical')):
                im=raw if axis=='original' else cv2.flip(raw,AXES[axis][0])
                m=mask if axis=='original' else cv2.flip(mask,AXES[axis][0])
                rgb=cv2.cvtColor(im,cv2.COLOR_GRAY2RGB)
                contours,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(rgb,contours,-1,(60,255,100),1,cv2.LINE_AA)
                panel.paste(Image.fromarray(rgb),(col*512,row*540+28))
                ImageDraw.Draw(panel).text((col*512+5,row*540+5),f'{axis} | parent roll {p["rotation_deg"]:.1f} deg',fill='white',font=font)
        panel.save(qa/f'mirror_triplets_{group}.jpg',quality=96)
    write(qa/'selected_parents.json',{'seed':SEED,'parent_ids':[p['id'] for p in parents],'note':NOTE})

def build():
    manifest,base_raw,base_rows,parents=load_inputs()
    preview(parents)
    out=NEW/'dataset';out.mkdir(parents=True,exist_ok=True)
    write(out/'BUILD_INCOMPLETE.json',{'status':'building_mirror_version'})
    for i,rel in enumerate(manifest['sha256']):
        target=out/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(BASE/rel,target)
        if (i+1)%4000==0:print(json.dumps({'copied_base_assets':i+1}),flush=True)
    provenance=out/'provenance';provenance.mkdir(exist_ok=True)
    for filename in ('manifest.json','preprocessing_config.json'):
        shutil.copyfile(BASE/filename,provenance/('base_'+filename))
    shutil.copyfile(BASE/'qa/checks.json',provenance/'base_qa_checks.json')
    children=[];axis_counts=Counter();source_axis_bins=defaultdict(set)
    for index,parent in enumerate(parents):
        arrays={k:cv2.imread(str(BASE/parent[k]),0) if parent.get(k) else None for k in ('image','lcd_mask','valid_mask','protected_mask')}
        for axis,(flipcode,F) in AXES.items():
            ident=f'lcd_mirror_{axis}_{index:04d}'
            child=copy.deepcopy(parent)
            child.update(id=ident,augmentation_kind='horizontal_mirror' if axis=='horizontal' else 'vertical_mirror',
                         is_augmented=True,mirrored=True,mirror_axis=axis,mirror_parent_id=parent['id'],
                         mirror_parent_image_sha256=manifest['sha256'][parent['image']],
                         rotation_semantics='parent_generating_roll_preserved_before_reflection',mirror_semantics=NOTE,
                         parent_to_sample_h=(F@np.asarray(parent['parent_to_sample_h'])).tolist(),
                         lcd_polygon_xy=project(parent['lcd_polygon_xy'],F).tolist(),
                         truncation_edges=swapped_edges(parent['truncation_edges'],axis))
            child['geometry']['mirrored']=True
            child['geometry']['mirror_axis']=axis
            child['geometry']['mirror_matrix']=F.tolist()
            child['geometry']['parent_geometry_semantics']='Original perspective/shear generation metadata precedes final reflection.'
            if 'full_footprint_xy' in parent['geometry']:
                child['geometry']['full_footprint_xy']=project(parent['geometry']['full_footprint_xy'],F).tolist()
            mirrored={}
            for key,arr in arrays.items():
                if arr is None:child[key]=None;continue
                flipped=cv2.flip(arr,flipcode)
                assert np.array_equal(cv2.flip(flipped,flipcode),arr)
                directory={'image':'images','lcd_mask':'lcd_masks','valid_mask':'valid_masks','protected_mask':'protected_masks'}[key]
                rel=f'{directory}/{ident}.png';(out/directory).mkdir(exist_ok=True)
                assert cv2.imwrite(str(out/rel),flipped,[cv2.IMWRITE_PNG_COMPRESSION,3])
                assert np.array_equal(cv2.imread(str(out/rel),0),flipped)
                child[key]=rel;mirrored[key]=flipped
            assert not np.any((mirrored['lcd_mask']>0)&(mirrored['valid_mask']==0))
            if 'protected_mask' in mirrored:
                assert not np.any((mirrored['protected_mask']>0)&((mirrored['valid_mask']==0)|(mirrored['lcd_mask']>0)))
            assert np.allclose(F@F,np.eye(3),rtol=0,atol=0)
            assert np.allclose(project(child['lcd_polygon_xy'],F),parent['lcd_polygon_xy'],rtol=0,atol=1e-10)
            assert np.allclose(F@np.asarray(child['parent_to_sample_h']),parent['parent_to_sample_h'],rtol=0,atol=1e-10)
            assert swapped_edges(child['truncation_edges'],axis)==parent['truncation_edges']
            children.append(child);axis_counts[axis]+=1
            source_axis_bins[(axis,parent['source_id'])].add(int(parent['rotation_deg']//30))
        if (index+1)%100==0:print(json.dumps({'mirrors_written':len(children),'total_mirrors':1200}),flush=True)
    assert all(len(b)==12 for b in source_axis_bins.values()) and len(source_axis_bins)==64
    assert axis_counts=={'horizontal':600,'vertical':600}
    with (out/'samples.jsonl').open('wb') as f:
        f.write(base_raw)
        for row in children:f.write((json.dumps(row,separators=(',',':'),allow_nan=False)+'\n').encode())
    all_rows=base_rows+children
    assert (out/'samples.jsonl').read_bytes()[:len(base_raw)]==base_raw
    for dirname,key in [('images','image'),('lcd_masks','lcd_mask'),('valid_masks','valid_mask'),('protected_masks','protected_mask')]:
        expected={r[key] for r in all_rows if r.get(key)}
        actual={p.relative_to(out).as_posix() for p in (out/dirname).iterdir() if p.is_file()}
        assert actual==expected,'Stale/missing image or mask assets'
    preserved=0
    for row in base_rows:
        for key in ('image','lcd_mask','valid_mask','protected_mask'):
            if row.get(key):assert sha(out/row[key])==manifest['sha256'][row[key]];preserved+=1
    config=json.loads((BASE/'preprocessing_config.json').read_text())
    config.update(mirroring=True,mirroring_scope='training_augmentation_only',runtime_mirroring=False,
                  mirror_axes=['horizontal','vertical'],mirror_semantics=NOTE)
    write(out/'preprocessing_config.json',config)
    checks={'status':'static_mirror_generation_checks_passed','dataset_version':'mirrored_v2',
            'base_manifest_sha256':BASE_SHA,'base_sample_count':4426,'mirror_parent_count':600,
            'sample_counts':dict(Counter(r['split'] for r in all_rows)),
            'mirror_counts':dict(axis_counts),'mirrored_training_fraction':.25,
            'train_mirror_counts_per_source_recording_and_axis':{axis:dict(Counter(r['source_group'] for r in children if r['mirror_axis']==axis)) for axis in AXES},
            'all32_train_families_cover_all12_parent_roll_bins_for_each_axis':True,
            'original_sample_lines_exact':True,'original_image_and_mask_bytes_preserved':preserved,
            'validation_test_rows_and_assets_unchanged':True,'exact_flip_pixels_and_roundtrip':True,
            'homography_polygon_and_truncation_roundtrip':True,'all_masks_inside_valid_and_protected_disjoint':True,
            'no_new_model_forward_training_or_video_test':True,'mirror_semantics':NOTE}
    write(out/'qa/checks.json',checks);write(NEW/'qa/build_checks.json',checks)
    for path in (NEW/'qa').glob('mirror_triplets_*.jpg'):shutil.copyfile(path,out/'qa'/path.name)
    write(provenance/'mirror_lineage.json',{'base_manifest_sha256':BASE_SHA,
          'base_samples_sha256':manifest['sha256']['samples.jsonl'],'parent_ids':[p['id'] for p in parents],
          'seed':SEED,'reflection_matrices':{a:f.tolist() for a,(_,f) in AXES.items()},'semantic_note':NOTE})
    (out/'BUILD_INCOMPLETE.json').unlink()
    updated=copy.deepcopy(manifest)
    updated.update(dataset_version='mirrored_v2',sample_count=5626,split_counts={'train':4800,'val':614,'test':212},
                   preprocessing=config,generator_sha256=sha(__file__),base_manifest_sha256=BASE_SHA,
                   mirror_policy=checks,opencv_version=cv2.__version__,numpy_version=np.__version__)
    updated['sha256']={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    write(out/'manifest.json',updated)
    print(json.dumps({'status':'complete','manifest_sha256':sha(out/'manifest.json'),'counts':updated['split_counts']}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['preview','build','package']);args=parser.parse_args()
    cv2.setNumThreads(2)
    if args.action=='preview':preview(load_inputs()[3])
    elif args.action=='build':build()
    else:
        from learned_lcd_v1.package_dataset import package
        package(NEW)
