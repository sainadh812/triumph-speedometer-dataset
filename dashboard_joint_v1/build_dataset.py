"""Build private, label-consistent video-derived augmentation datasets. No ML.

This version augments reviewed real needle/bar states and can add explicitly
marked procedural RPM-bar renders. Their scalar is a rendering target, never
a physical engine measurement. All descendants retain their source time-block
split. Do not publish the resulting private archive.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
try:
    from .preprocess import SIZES, POINT_NAMES, project, letterbox_matrix, dashboard_matrix, rpm_matrix, rpm_roi_corners, rpm_from_roi_matrix, warp_input, map_labels
except ImportError:
    from preprocess import SIZES, POINT_NAMES, project, letterbox_matrix, dashboard_matrix, rpm_matrix, rpm_roi_corners, rpm_from_roi_matrix, warp_input, map_labels

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def augment_geometry(rng, size, task, variant):
    w,h=size
    if variant == 0:
        return np.eye(3), {'identity':True}
    angle = float(rng.uniform(-30,30))
    if variant % 8 == 0 and task != 'rpm':
        angle = float(rng.uniform(-180,180))
    if task == 'rpm':
        angle = float(rng.uniform(-5,5))
    scale = float(rng.uniform(.80,1.04) if task=='localizer' else rng.uniform(.91,1.02))
    a=np.vstack([cv2.getRotationMatrix2D((w/2,h/2),angle,scale),[0,0,1]])
    a[:2,2] += rng.uniform(-.035,.035,2)*[w,h]
    shear = float(rng.uniform(-.06,.06) if task != 'rpm' else rng.uniform(-.025,.025))
    s=np.array([[1,shear,-shear*h/2],[0,1,0],[rng.uniform(-.00008,.00008),rng.uniform(-.00008,.00008),1]])
    a=s@a
    flip = 'none'
    if task!='rpm' and variant % 12 in (10,11):
        if variant%12==10:
            a=np.array([[-1,0,w-1],[0,1,0],[0,0,1]])@a
            flip='horizontal'
        else:
            a=np.array([[1,0,0],[0,-1,h-1],[0,0,1]])@a
            flip='vertical'
    return a, {'roll_degrees':angle,'scale':scale,'shear':shear,'flip':flip}

def photometric(img,rng,variant):
    if variant % 4 == 0:
        return img, {'identity':True}
    already_soft = cv2.Laplacian(img,cv2.CV_32F).var()<80
    gamma=float(rng.uniform(.75,1.35)); gain=float(rng.uniform(.75,1.20))
    out=(img.astype(np.float32)/255)**gamma*255*gain+rng.uniform(-12,12)
    glare = bool(rng.random()<.2)
    if glare:
        yy,xx=np.mgrid[:img.shape[0],:img.shape[1]]
        cx,cy=rng.uniform(0,img.shape[1]),rng.uniform(0,img.shape[0])
        sigma=rng.uniform(.1,.35)*max(img.shape)
        out+=rng.uniform(8,38)*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*sigma*sigma))
    sigma=float(rng.uniform(0,3.5))
    out=np.clip(out+rng.normal(0,sigma,out.shape),0,255).astype(np.uint8)
    blur='none'
    if not already_soft and rng.random()<.2:
        k=3; ker=np.zeros((k,k),np.float32)
        if rng.random()<.5: ker[k//2,:]=1/k; blur='motion_x'
        else: ker[:,k//2]=1/k; blur='motion_y'
        out=cv2.filter2D(out,-1,ker)
    elif not already_soft and rng.random()<.12:
        out=cv2.GaussianBlur(out,(3,3),.6); blur='gaussian'
    quality=int(rng.integers(88 if already_soft else 78,97))
    ok,buf=cv2.imencode('.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,quality])
    if ok: out=cv2.imdecode(buf,cv2.IMREAD_GRAYSCALE)
    return out, {'gamma':gamma,'gain':gain,'noise_sigma':sigma,'soft_glare':glare,'blur':blur,'jpeg_quality':quality}

def reviewed_rpm_roi(row):
    explicit=row.get('rpm_roi_corners')
    if explicit is None: return None
    review=row.get('rpm_roi_review',{})
    if review.get('reviewed') is not True or not str(review.get('reason','')).strip():
        raise ValueError('Explicit RPM crop geometry requires a review and reason')
    rpm_matrix(explicit) # finite, convex, nondegenerate semantic quad
    return np.asarray(explicit,np.float32)

def rpm_crop_geometry(row):
    """Choose reviewed context corners even if the far side of the LCD is cut off."""
    roi=reviewed_rpm_roi(row)
    if roi is not None: return roi, rpm_from_roi_matrix
    corners=row.get('lcd_rectification_corners') or row.get('lcd_corners')
    if not corners or any(p is None for p in corners): return None, None
    rpm_matrix(corners)
    return np.asarray(corners,np.float32), rpm_matrix

def annotation_targets(row,task,source_shape=None):
    """Return geometry, visibility and whether visibility was actually reviewed.

    Being unable to read RPM digits does not establish that the RPM crop is
    absent. Explicit reviewed crop geometry is independent of numeric labels.
    """
    if task=='localizer':
        corners=row.get('lcd_corners',[])
        corners=(corners+[None]*4)[:4]
        vis=row.get('lcd_corner_visibility',row.get('lcd_corner_visible',[p is not None for p in corners]))
        points=[p if p is not None else [0,0] for p in corners]
        quad=row.get('lcd_rectification_corners') or row.get('lcd_corners')
        roi=np.zeros((4,2),np.float32);roi_vis=[False]*4;roi_known=[False]*4
        justified=sum(v.get('visible',False) for v in row.get('rpm_points',{}).values())>=3
        explicit=reviewed_rpm_roi(row)
        if explicit is not None:
            roi=explicit
            roi_known=[True]*4
        elif justified and quad and len(quad)==4 and all(p is not None for p in quad):
            rpm_matrix(quad) # validate nondegenerate semantic geometry first
            roi=rpm_roi_corners(quad)
            roi_known=[True]*4
        if all(roi_known):
            if source_shape is None: raise ValueError('Source shape required for RPM crop visibility')
            hh,ww=source_shape[:2]
            roi_vis=((roi[:,0]>=0)&(roi[:,0]<=ww-1)&(roi[:,1]>=0)&(roi[:,1]<=hh-1)).tolist()
        return np.array(points+roi.tolist(),np.float32),list(vis)+roi_vis,[True]*4+roi_known
    points=row[task+'_points']
    values=[points.get(n,{}) for n in POINT_NAMES[task]]
    visible=[bool(v.get('visible',False) and v.get('xy') is not None) for v in values]
    known=[bool(v.get('visibility_known',True)) for v in values]
    if any(v and not k for v,k in zip(visible,known)):
        raise ValueError('Visible landmarks must have known visibility')
    return np.array([v.get('xy') or [0,0] for v in values],np.float32),visible,known

def annotation_points(row,task,source_shape=None):
    # Keep the geometry-only API for audits and earlier callers.
    points,visible,_=annotation_targets(row,task,source_shape)
    return points,visible

def seed_for(name):
    return int(hashlib.sha256(name.encode()).hexdigest()[:16],16)

def make_dataset(source_root,output,train_variants=48,val_variants=4,synthetic_rpm_count=0):
    source_root,output=Path(source_root),Path(output)
    if output.exists():
        raise FileExistsError('Choose a new output folder; existing datasets are immutable.')
    inventory=read(source_root/'source_inventory.json')['sources']
    for extra in sorted(source_root.glob('source_inventory_*_extra.json')):
        content=read(extra); inventory+=content.get('sources',content) if isinstance(content,dict) else content
    by_id={s['id']:s for s in inventory}
    if len(by_id)!=len(inventory): raise ValueError('Duplicate source IDs')
    annotations=[]
    for name in ('annotations_day.json','annotations_night.json'):
        annotations+=read(source_root/name)['sources']
    if len({a['id'] for a in annotations})!=len(annotations): raise ValueError('Duplicate annotation IDs')
    for row in annotations:
        if row.get('annotation_status','reviewed')!='reviewed':
            raise ValueError('Unreviewed annotation proposal cannot enter training: '+row['id'])
    samples={t:[] for t in SIZES}; skipped=[]; coverage=[]
    output.mkdir(parents=True)
    for task in SIZES:
        (output/task/'images').mkdir(parents=True)
        if task=='localizer': (output/task/'masks').mkdir()
    for row in annotations:
        meta=by_id[row['id']]; split=meta['split']; image=cv2.imread(str(source_root/meta['image']))
        if image is None: raise ValueError('Missing source '+row['id'])
        h,w=image.shape[:2]
        polygon=row.get('dashboard_polygon',[]); lcd=row.get('lcd_polygon',[])
        present=int(row.get('present',bool(polygon)))
        variants=train_variants if split=='train' else val_variants if split=='val' else 1
        for task,size in SIZES.items():
            points,visible,visibility_known=annotation_targets(row,task,image.shape)
            if task=='speed' and (not present or sum(visible)<3):
                skipped.append({'id':row['id'],'task':task,'reason':'insufficient visible speed supervision'});continue
            if task=='rpm':
                corners,crop_matrix=rpm_crop_geometry(row)
                if corners is None or sum(visible)<3:
                    skipped.append({'id':row['id'],'task':task,'reason':'no justified quad or insufficient visible RPM supervision'});continue
                baseline=crop_matrix(corners)
                mapped,mappedvis=map_labels(points,visible,baseline,size)
                # Every visible anchor must fit. Never silently train a clipped scale.
                if sum(mappedvis)<sum(visible):
                    skipped.append({'id':row['id'],'task':task,'reason':'fixed RPM band excludes visible landmark; requires geometry review'});continue
                coverage.append({'id':row['id'],'visible_rpm_points':int(sum(visible)),
                                 'min_xy':mapped[mappedvis].min(0).tolist(),'max_xy':mapped[mappedvis].max(0).tolist()})
            for variant in range(variants):
                ident=f'{row["id"]}_{task}_{variant:03d}'; rng=np.random.default_rng(seed_for(ident))
                if task=='localizer': base=letterbox_matrix([0,0,w,h],size)
                elif task=='speed':
                    jitter=(0,0,1) if variant==0 else (*rng.uniform(-.025,.025,2),float(rng.uniform(.96,1.08)))
                    base=dashboard_matrix(polygon,image.shape,jitter=jitter)
                else:
                    perturb=None if variant==0 else rng.normal(0,.007,(4,2))*np.linalg.norm(np.asarray(corners)[1]-np.asarray(corners)[0])
                    base=crop_matrix(corners if perturb is None else corners+perturb)
                aug,geo=augment_geometry(rng,size,task,variant); matrix=aug@base
                out,mask=warp_input(image,matrix,task,lcd)
                out,photo=photometric(out,rng,variant)
                if task=='speed': out[mask>0]=128
                pts,vis=map_labels(points,visible,matrix,size)
                if task=='speed':
                    for idx in np.where(vis)[0]:
                        x,y=np.round(pts[idx]).astype(int)
                        if mask[y,x]>0:
                            vis[idx]=False;pts[idx]=0
                if task=='rpm' and sum(vis)<sum(visible): continue
                if task=='speed' and sum(vis)<3: continue
                item={'id':ident,'source_id':row['id'],'split':split,'image':f'images/{ident}.png',
                      'width':size[0],'height':size[1],'points':pts.tolist(),'visible':vis.tolist(),
                      'visibility_known':list(visibility_known),
                      'lighting':meta['lighting'],'state':row.get('state','uncertain'),
                      'source_time_seconds':meta['time_seconds'],'transform':matrix.tolist(),
                      'augmentation':{'geometry':geo,'appearance':photo}}
                if task=='localizer':
                    pp=project(polygon,matrix) if present else np.empty((0,2))
                    if len(pp):
                        lo=np.clip(pp.min(0),[0,0],size);hi=np.clip(pp.max(0),[0,0],size)
                        box=[*lo,*hi]
                    else: box=[0,0,0,0]
                    item.update({'lcd_mask':f'masks/{ident}.png','dashboard_box':[float(v) for v in box],
                                 'present':int(present and box[2]>box[0] and box[3]>box[1])})
                    if not item['present']:
                        mask[:]=0;item['points']=[[0,0]]*8;item['visible']=[False]*8
                        item['visibility_known']=[True]*8
                    cv2.imwrite(str(output/task/item['lcd_mask']),mask)
                cv2.imwrite(str(output/task/item['image']),out)
                samples[task].append(item)
        # Verified dashboard-free background crops supply explicit negatives.
        if split=='train' and polygon:
            protect=np.zeros((h,w),np.uint8)
            cv2.fillPoly(protect,[np.round(polygon).astype(np.int32)],255)
            for corner,(x0,y0) in enumerate(((0,0),(w//2,0),(0,h//2),(w//2,h//2))):
                box=[x0,y0,min(w,x0+w//2),min(h,y0+h//2)]
                patch=protect[box[1]:box[3],box[0]:box[2]]
                if patch.mean()>0: continue
                for v in range(4):
                    ident=f'{row["id"]}_negative_{corner}_{v}';rng=np.random.default_rng(seed_for(ident))
                    cropped=image[box[1]:box[3],box[0]:box[2]]
                    ch,cw=cropped.shape[:2]
                    matrix=letterbox_matrix([0,0,cw,ch],SIZES['localizer']);out,_=warp_input(cropped,matrix,'localizer')
                    out,_=photometric(out,rng,v)
                    item={'id':ident,'source_id':row['id'],'split':split,'image':f'images/{ident}.png',
                          'lcd_mask':f'masks/{ident}.png','width':512,'height':512,'points':[[0,0]]*8,
                          'visible':[False]*8,'visibility_known':[True]*8,'present':0,'dashboard_box':[0,0,0,0],'negative':True,
                          'lighting':meta['lighting']}
                    cv2.imwrite(str(output/'localizer'/item['image']),out)
                    cv2.imwrite(str(output/'localizer'/item['lcd_mask']),np.zeros((512,512),np.uint8))
                    samples['localizer'].append(item)
    if synthetic_rpm_count:
        try:
            from .synthetic_rpm import generate_examples
        except ImportError:
            from synthetic_rpm import generate_examples
        master=next(r for r in annotations if r['id']=='night_00')
        meta=by_id['night_00']
        original=cv2.imread(str(source_root/meta['image']))
        for index,example in enumerate(generate_examples(original,master,source_split=meta['split'],count=synthetic_rpm_count)):
            ident=f'night_00_rpm_render_{index:03d}';rng=np.random.default_rng(seed_for(ident))
            matrix,geometry=augment_geometry(rng,SIZES['rpm'],'rpm',index)
            pts,vis=map_labels(example['points'],example['visible'],matrix,SIZES['rpm'])
            if int(sum(vis))!=sum(example['visible']):continue
            image=cv2.warpPerspective(example['image'],matrix,SIZES['rpm'],borderValue=128)
            image=cv2.GaussianBlur(image,(3,3),.65) # soften perfectly regular rendered LCD stripes
            image,appearance=photometric(image,rng,index)
            row={k:v for k,v in example.items() if k not in ('image','points','visible','canonical_fill_mask')}
            row.update(id=ident,image=f'images/{ident}.png',width=256,height=384,points=pts.tolist(),visible=vis.tolist(),visibility_known=[True]*len(vis),
                       lighting='night',state='synthetic_display',source_time_seconds=meta['time_seconds'],
                       augmentation={'geometry':geometry,'appearance':appearance},transform=matrix.tolist())
            cv2.imwrite(str(output/'rpm'/row['image']),image)
            samples['rpm'].append(row)
    manifest={'schema_version':2,'dataset_id':output.name,'created_utc':datetime.now(timezone.utc).isoformat(),
              'visibility_policy':'Unknown visibility is excluded from visibility loss and metrics. Reviewed RPM crop geometry is independent of readable RPM anchors.',
              'description':'Reviewed real states with deterministic augmentation; additional RPM bar renders explicitly marked synthetic, not physical measurements.',
              'split_policy':'Time-block development splits within two recordings. All descendants stay with source. Not unseen-recording evaluation.',
              'source_count':len(annotations),'synthetic_rpm_examples':sum(bool(r.get('synthetic')) for r in samples['rpm']),
              'synthetic_speed_positions':False,'tasks':{},'files':[]}
    for task,rows in samples.items():
        counts=dict(Counter(x['split'] for x in rows))
        if any(counts.get(k,0)==0 for k in ('train','val','test')):
            raise ValueError(f'{task} missing development split: {counts}')
        write(output/task/'samples.json',{'point_names':POINT_NAMES[task],'input_size':list(SIZES[task]),'samples':rows})
        manifest['tasks'][task]={'input_size':list(SIZES[task]),'point_names':POINT_NAMES[task], 'counts':counts,
                                'source_counts':dict(Counter(by_id[s]['split'] for s in {x['source_id'] for x in rows})),
                                'visible_counts':dict(zip(POINT_NAMES[task],np.asarray([r['visible'] for r in rows]).sum(0).astype(int).tolist())),
                                'unknown_visibility_counts':dict(zip(POINT_NAMES[task],(~np.asarray([r['visibility_known'] for r in rows],bool)).sum(0).astype(int).tolist()))}
    # Private provenance stays in private archive. Exclude the raw videos.
    write(output/'provenance'/'annotations.json',{'sources':annotations})
    write(output/'provenance'/'source_inventory.json',{'sources':inventory})
    write(output/'qa.json',{'skipped':skipped,'rpm_crop_coverage':coverage,'status':'requires visual inspection before upload'})
    for path in sorted(output.rglob('*')):
        if path.is_file():manifest['files'].append({'path':path.relative_to(output).as_posix(),'sha256':digest(path),'size':path.stat().st_size})
    write(output/'dataset_manifest.json',manifest)
    return manifest

def package(root):
    root=Path(root);target=root.with_suffix('.zip')
    if target.exists():raise FileExistsError(target)
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for file in sorted(root.rglob('*')):
            if file.is_file():z.write(file,file.relative_to(root).as_posix())
    return {'path':str(target),'sha256':digest(target),'size':target.stat().st_size}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sources',required=True);p.add_argument('--output',required=True)
    p.add_argument('--train-variants',type=int,default=48);p.add_argument('--val-variants',type=int,default=4)
    p.add_argument('--synthetic-rpm-count',type=int,default=0)
    p.add_argument('--zip',action='store_true');a=p.parse_args()
    m=make_dataset(a.sources,a.output,a.train_variants,a.val_variants,a.synthetic_rpm_count)
    print(json.dumps({k:v['counts'] for k,v in m['tasks'].items()}))
    if a.zip:print(json.dumps(package(a.output)))
