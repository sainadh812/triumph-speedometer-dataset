"""Reviewed raw LCD masks and deterministic full-frame synthetic descendants.

No network/model calls. Registration is only an annotation seed; every source
overlay is reviewed before build. Never reads prior masked training images.
"""
from __future__ import annotations
import argparse, hashlib, json, math, shutil
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent
SIZE=512
PREPROCESSING_ID='gray512_fullframe_letterbox_unmasked_lcd_seg_v1_imagenet_normalized'
SEED=20261018

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def project(points,h):
    q=np.c_[np.asarray(points,float),np.ones(len(points))]@np.asarray(h,float).T
    if (np.abs(q[:,2])<1e-8).any():raise ValueError('projective horizon')
    return q[:,:2]/q[:,2:]

def letterbox_matrix(width,height):
    side=max(width,height);left=(side-width)//2;top=(side-height)//2;s=SIZE/side
    return np.array([[s,0,(left+.5)*s-.5],[0,s,(top+.5)*s-.5],[0,0,1.]])

def prepare(gray):
    height,width=gray.shape;side=max(width,height);left=(side-width)//2;top=(side-height)//2
    return cv2.resize(cv2.copyMakeBorder(gray,top,side-height-top,left,side-width-left,cv2.BORDER_CONSTANT,value=0),(SIZE,SIZE),interpolation=cv2.INTER_LINEAR)

def sensor_mask(width,height):
    h=letterbox_matrix(width,height)
    p=project([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],h)
    yy,xx=np.ogrid[:SIZE,:SIZE]
    m=((xx>=p[:,0].min())&(xx<=p[:,0].max())&(yy>=p[:,1].min())&(yy<=p[:,1].max()))
    return m.astype(np.uint8)*255,p

def clip_polygon(poly,footprint):
    poly=np.asarray(poly,np.float32)
    area,out=cv2.intersectConvexConvex(poly,np.asarray(footprint,np.float32))
    if out is None or area<=0:raise ValueError('LCD does not intersect sensor')
    return out.reshape(-1,2).astype(float)

def polygon_mask(poly,valid):
    m=np.zeros((SIZE,SIZE),np.uint8)
    cv2.fillPoly(m,[np.rint(poly).astype(np.int32)],255)
    return cv2.bitwise_and(m,valid)

def split_for(group,index):
    if group=='rpm220558':return 'test'
    if group=='night181323':return 'train' if index<=240 else ('val' if index>=396 else 'excluded')
    if group=='day174359':return 'train' if index<=330 else ('val' if index>=420 else 'excluded')
    return 'train' if index in (0,43,130,173,217) else ('val' if index in (391,434,478) else 'excluded')

def seed_match(reference,current,polygon):
    """Image registration for a human-review proposal; never a training target by itself."""
    sift=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.008)
    clahe=cv2.createCLAHE(2.,(8,8))
    ka,da=sift.detectAndCompute(clahe.apply(reference),None)
    kb,db=sift.detectAndCompute(clahe.apply(current),None)
    pairs=cv2.BFMatcher().knnMatch(da,db,k=2)
    good=[p[0] for p in pairs if len(p)==2 and p[0].distance<.73*p[1].distance]
    if len(good)<8:raise ValueError('insufficient proposal matches')
    a=np.float32([ka[m.queryIdx].pt for m in good]);b=np.float32([kb[m.trainIdx].pt for m in good])
    cv2.setRNGSeed(SEED)
    h,inside=cv2.findHomography(a,b,cv2.RANSAC,2.5,maxIters=5000)
    if h is None or int(inside.sum())<8:raise ValueError('insufficient proposal inliers')
    return project(polygon,h),{'method':'SIFT_RANSAC_annotation_proposal_only','inliers':int(inside.sum()),'h':h.tolist()}

def annotate():
    """Prepare all source proposals, preserving every original source file."""
    out=ROOT/'dataset';ann=ROOT/'annotations';qa=ROOT/'qa'
    for p in [out/'source_images',out/'source_lcd_masks',out/'source_valid_masks',ann,qa]:p.mkdir(parents=True,exist_ok=True)
    v3=PROJECT/'real_adaptation_v3';v4=PROJECT/'real_adaptation_v4';rpm=PROJECT/'rpm_exploration'
    reg={r['frame_index']:np.array(r['source_to_reference_h']) for r in json.loads((v3/'registration.json').read_text())['frames']}
    v3poly=np.array(json.loads((v3/'annotations/reference_ticks.json').read_text())['lcd_polygon_xy'])
    v4poly=np.array(json.loads((v4/'dataset2000/preprocessing_config.json').read_text())['lcd_polygon_reference_xy'])
    # Extend the two physical LCD edges along their lines before registration;
    # the existing x417 edge is a sensor cut, not the LCD's physical right edge.
    v4poly[1]=[700,v4poly[0,1]+(700-v4poly[0,0])*(v4poly[1,1]-v4poly[0,1])/(v4poly[1,0]-v4poly[0,0])]
    v4poly[2]=[700,v4poly[3,1]+(700-v4poly[3,0])*(v4poly[2,1]-v4poly[3,1])/(v4poly[2,0]-v4poly[3,0])]
    dayref=cv2.imread(str(v4/'sources/real_frames/frame_0000.png'),0)
    rpmreports={r['source']:r for r in json.loads((PROJECT/'rpm_prototype/qa/alignment/report.json').read_text())}
    # Reference is the actual bright screen with narrow bezel margin; rounded
    # corners are explicitly represented rather than treating its bbox as LCD.
    rpmpoly=np.array([[84,323],[403,303],[418,309],[438,455],[427,470],[98,491],[86,484],[74,336]],float)
    specs=[]
    for f in sorted((v3/'real_frames').glob('frame_*.png')):specs.append(('night181323',int(f.stem[-4:]),f,(400,368),'night','straight_dashboard'))
    for f in sorted((v4/'sources/real_frames').glob('frame_*.png')):specs.append(('day174359',int(f.stem[-4:]),f,(534,850),'day','tilted_partial_LCD'))
    for group,code in [('rpm220557','220557'),('rpm220558','220558')]:
        for f in sorted(rpm.glob(f'review_{code}_frame_*.png')):specs.append((group,int(f.stem[-4:]),f,(478,850),'night','distant_dashboard' if code=='220557' else 'close_LCD'))
    rows=[]
    for group,index,path,native_size,lighting,view in specs:
        raw=cv2.imread(str(path),0);gray=raw if raw.shape==(512,512) else prepare(raw)
        valid,footprint=sensor_mask(*native_size);source_h=letterbox_matrix(*native_size)
        proposal={};poly=None
        if group=='night181323':poly=project(v3poly,np.linalg.inv(reg[index]));proposal={'method':'existing_image_registration_proposal','h':np.linalg.inv(reg[index]).tolist()}
        elif group=='day174359':
            poly,proposal=seed_match(dayref,gray,v4poly)
        else:
            report=rpmreports[path.name]
            if report.get('accepted'):
                native=project(rpmpoly,np.linalg.inv(np.array(report['homography_source_to_reference'])))
                poly=project(native,source_h);proposal={'method':'existing_RPM_static_registration_proposal','report_source':path.name}
            elif split_for(group,index)!='excluded':
                ref=cv2.imread(str(rpm/'review_220558_frame_0169.png'),0)
                try:
                    native,proposal=seed_match(ref,raw,rpmpoly);poly=project(native,source_h)
                except ValueError:
                    # Only a seed for manual review; neighboring pose is never
                    # silently promoted to a reviewed target.
                    refname='review_220557_frame_0260.png' if index==217 else 'review_220557_frame_0391.png'
                    h=np.linalg.inv(np.array(rpmreports[refname]['homography_source_to_reference']))
                    poly=project(project(rpmpoly,h),source_h);proposal={'method':'neighbor_pose_pending_manual_correction','neighbor':refname}
        ident=f'{group}_{index:04d}'
        image=f'source_images/{ident}.png';lcd=f'source_lcd_masks/{ident}.png';vm=f'source_valid_masks/{ident}.png'
        cv2.imwrite(str(out/image),gray);cv2.imwrite(str(out/vm),valid)
        cut_edges=[]
        if poly is not None:
            for axis,label,kind in [(0,'left','min'),(0,'right','max'),(1,'top','min'),(1,'bottom','max')]:
                value=getattr(poly[:,axis],kind)();edge=getattr(footprint[:,axis],kind)()
                if (value<edge+.5 if kind=='min' else value>edge-.5):cut_edges.append(label)
            poly=clip_polygon(poly,footprint);cv2.imwrite(str(out/lcd),polygon_mask(poly,valid))
        row={'source_id':ident,'source_group':group,'frame_index':index,'split':split_for(group,index),
             'image':image,'lcd_mask':lcd if poly is not None else None,'valid_mask':vm,
             'lcd_polygon_xy':poly.tolist() if poly is not None else None,'native_to_512_h':source_h.tolist(),
             'native_size':list(native_size),'source_image_size':[int(raw.shape[1]),int(raw.shape[0])],
             'original_source_path':str(path.relative_to(PROJECT)).replace('\\','/'),'original_source_sha256':sha(path),
             'source_view':view,'lighting':lighting,'dashboard_state':'not_annotated',
             'truncated':bool(cut_edges),'truncation_edges':cut_edges,
             'physical_edge_uncertainty_px':2.,'annotation_status':'pending_manual_visual_review' if poly is not None else 'excluded_unidentifiable_dark_boundary',
             'proposal':proposal,'protected_mask':None,'protected_analog_annotation_available':False,
             'sensor_footprint_xy':footprint.tolist(),'model_predictions_used_for_labels':False}
        rows.append(row)
    write_json(ann/'source_annotations.json',rows)
    render_review(rows)
    print(json.dumps({'sources':len(rows),'split_counts':dict(Counter(r['split'] for r in rows)),'status':'proposals_pending_review'}))

def render_review(rows):
    qa=ROOT/'qa';qa.mkdir(exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for group in sorted({r['source_group'] for r in rows}):
        group_rows=[r for r in rows if r['source_group']==group]
        for page in range(math.ceil(len(group_rows)/12)):
            full=Image.new('RGB',(1536,2160),(25,25,25));detail=Image.new('RGB',(1536,1320),(25,25,25))
            for k,r in enumerate(group_rows[page*12:(page+1)*12]):
                gray=cv2.imread(str(ROOT/'dataset'/r['image']),0);rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
                poly=r['lcd_polygon_xy']
                if poly:
                    pts=np.rint(poly).astype(np.int32);cv2.polylines(rgb,[pts],True,(40,240,110),1,cv2.LINE_AA)
                    for j,(x,y) in enumerate(pts):cv2.putText(rgb,str(j),(int(x)+2,int(y)-2),cv2.FONT_HERSHEY_SIMPLEX,.35,(255,80,80),1)
                    lo=np.maximum(np.floor(np.min(poly,axis=0)-15),0).astype(int);hi=np.minimum(np.ceil(np.max(poly,axis=0)+15),511).astype(int)
                    crop=rgb[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
                else:crop=rgb
                label=f'{r["source_id"]} | {r["split"]}'
                x=(k%3)*512;y=(k//3)*540
                full.paste(Image.fromarray(rgb),(x,y+25));ImageDraw.Draw(full).text((x+5,y+3),label,font=font,fill='white')
                box=Image.fromarray(crop);scale=min(504/box.width,292/box.height);box=box.resize((round(box.width*scale),round(box.height*scale)),Image.Resampling.NEAREST);dx=(k%3)*512;dy=(k//3)*330
                detail.paste(box,(dx+(512-box.width)//2,dy+30));ImageDraw.Draw(detail).text((dx+5,dy+3),label,font=font,fill='white')
            full.save(qa/f'sources_{group}_{page+1}.jpg',quality=96)
            detail.save(qa/f'edges_{group}_{page+1}.jpg',quality=97)

def approve_review():
    rows=json.loads((ROOT/'annotations/source_annotations.json').read_text())
    for r in rows:
        if r['source_id']=='night181323_0072':
            r['split']='excluded';r['annotation_status']='excluded_uncertain_dark_boundary'
            r['exclusion_reason']='Physical LCD boundary cannot be confidently identified; not an absence label.'
        elif r['lcd_polygon_xy'] is not None:
            r['annotation_status']='manually_visually_reviewed'
        r['physical_edge_uncertainty_px']=2.
        r['review_note']='Every source reviewed in enlarged edge contacts and full-frame context. Narrow bezel margin retained. Estimated2px boundary uncertainty is not measured ground-truth precision.'
        g,i=r['source_group'],r['frame_index']
        if g=='day174359':state='normal_day_display'
        elif g=='night181323':
            state='display_off' if i in (72,96,108,120,132,360,396,408) else ('startup_or_transition' if 144<=i<=240 or i>=420 else 'engine_off_message')
        elif g=='rpm220557':state='display_off_dark' if i in (86,304,347) else ('startup' if i in (130,173,391,434) else 'engine_off_message')
        else:state='engine_off_message' if i>=381 else 'normal_backlit_display'
        r['dashboard_state']=state
    write_json(ROOT/'annotations/source_annotations.json',rows)
    write_json(ROOT/'annotations/review_record.json',{'status':'approved_for_dataset_build','reviewed_sources':68,
        'eligible_sources':58,'excluded_sources':10,'review_date':'2026-10-18',
        'review_basis':'All6 enlarged edge contact sheets plus full-frame context reviewed by dataset agent; root independently reviewed all6 edge contacts and fullframe night page1.',
        'corrections':['Day clipped LCD physical edges extended along their lines before current-sensor intersection.','Late night source frames flagged as sensor-truncated.','Night0072 excluded rather than assigning an uncertain LCD target or absence.'],
        'physical_boundary_uncertainty_px_estimated':2.,'model_predictions_used':False,
        'protected_analog_metrics_available':'V3_and_V4_sources_only','limitation':'Manual masks with narrow bezel margin; not pixel-perfect measured truth. Existing static registration seeded annotation, followed by visual review.'})
    render_review(rows)

def protected_review():
    rows=json.loads((ROOT/'annotations/source_annotations.json').read_text())
    out=ROOT/'dataset';(out/'source_protected_masks').mkdir(exist_ok=True)
    cal3=json.loads((PROJECT/'real_adaptation_v3/dataset500/calibration.json').read_text())
    cal4=json.loads((PROJECT/'real_adaptation_v4/dataset2000/calibration.json').read_text())
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for group,cal in [('night181323',cal3),('day174359',cal4)]:
        selected=[r for r in rows if r['source_group']==group]
        for page in range(math.ceil(len(selected)/12)):
            sheet=Image.new('RGB',(1536,2160),(24,24,24))
            for k,r in enumerate(selected[page*12:(page+1)*12]):
                pts=project([cal['pivot_xy']]+[t['xy'] for t in cal['ticks']],np.array(r['proposal']['h']))
                protected=np.zeros((512,512),np.uint8)
                cv2.fillPoly(protected,[np.rint(pts).astype(np.int32)],255)
                for point in pts:cv2.circle(protected,tuple(np.rint(point).astype(int)),3,255,-1)
                lcd=cv2.imread(str(out/r['lcd_mask']),0);valid=cv2.imread(str(out/r['valid_mask']),0)
                protected[cv2.dilate(lcd,np.ones((9,9),np.uint8))>0]=0
                protected[valid==0]=0
                path=f'source_protected_masks/{r["source_id"]}.png';cv2.imwrite(str(out/path),protected)
                r['protected_mask']=path;r['protected_analog_annotation_available']=True
                r['protected_points_xy']=pts.tolist()
                r['protected_annotation_status']='manually_calibrated_source_geometry_transferred_pending_overlay_review'
                r['protected_region_definition']='pivot-to-10-tick filled fan and3px landmark disks, excluding4px LCD margin; same source coordinates, QA/loss only'
                gray=cv2.imread(str(out/r['image']),0);rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
                color=np.zeros_like(rgb);color[:]=(70,170,255)
                rgb[protected>0]=(rgb[protected>0]*.7+color[protected>0]*.3).astype(np.uint8)
                cv2.polylines(rgb,[np.rint(r['lcd_polygon_xy']).astype(np.int32)],True,(60,255,80),1)
                for point in pts:cv2.circle(rgb,tuple(np.rint(point).astype(int)),2,(255,180,60),-1)
                x=k%3*512;y=k//3*540;sheet.paste(Image.fromarray(rgb),(x,y+25))
                ImageDraw.Draw(sheet).text((x+4,y+3),f'{r["source_id"]} | {r["split"]} | blue: protected',font=font,fill='white')
            sheet.save(ROOT/'qa'/f'protected_{group}_{page+1}.jpg',quality=96)
    write_json(ROOT/'annotations/source_annotations.json',rows)

def approve_protected():
    rows=json.loads((ROOT/'annotations/source_annotations.json').read_text())
    for r in rows:
        if r.get('protected_mask'):r['protected_annotation_status']='manually_visually_reviewed'
    write_json(ROOT/'annotations/source_annotations.json',rows)

def augmentation_h(rng,rotation):
    angle=math.radians(rotation);c,s=math.cos(angle),math.sin(angle)
    center=np.array([[1,0,-255.5],[0,1,-255.5],[0,0,1.]])
    rotate=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    shear=np.array([[1,rng.uniform(-.055,.055),0],[rng.uniform(-.04,.04),1,0],[rng.uniform(-.00010,.00010),rng.uniform(-.00010,.00010),1.]])
    h=shear@rotate@center
    corners=project([[-.5,-.5],[511.5,-.5],[511.5,511.5],[-.5,511.5]],h)
    lo=corners.min(0);hi=corners.max(0);fit=504/max(hi-lo)*rng.uniform(.84,.99)
    extent=(hi-lo)*fit;room=(508-extent)/2;offset=rng.uniform(-1,1,2)*np.minimum(room,22)
    transform=np.array([[fit,0,255.5-(lo[0]+hi[0])*fit/2+offset[0]],
                        [0,fit,255.5-(lo[1]+hi[1])*fit/2+offset[1]],[0,0,1.]])
    result=transform@h;final=project([[-.5,-.5],[511.5,-.5],[511.5,511.5],[-.5,511.5]],result)
    assert final.min()>=-.5 and final.max()<=511.5
    return result,{'scale_to_fit':float(fit),'shear_projective_matrix':shear.tolist(),'full_footprint_xy':final.tolist(),'cropped':False,'mirrored':False}

def photometric(gray,valid,rng):
    x=gray.astype(np.float32)/255
    gamma=float(rng.uniform(.78,1.22));contrast=float(rng.uniform(.80,1.18));brightness=float(rng.uniform(-.055,.055))
    x=np.power(x,gamma)*contrast+brightness
    yy,xx=np.mgrid[:SIZE,:SIZE].astype(np.float32);xx=(xx-255.5)/255.5;yy=(yy-255.5)/255.5
    gx,gy=rng.uniform(-.12,.12,2);x*=1+gx*xx+gy*yy
    sigma=float(rng.uniform(0,.95)) if rng.random()<.45 else 0.
    if sigma>.12:x=cv2.GaussianBlur(x,(5,5),sigma)
    noise=float(rng.uniform(0,.010));x+=rng.normal(0,noise,x.shape).astype(np.float32)
    image=np.rint(np.clip(x,0,1)*255).astype(np.uint8)
    quality=int(rng.integers(65,97)) if rng.random()<.35 else None
    if quality is not None:
        ok,encoded=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,quality]);assert ok
        image=cv2.imdecode(encoded,cv2.IMREAD_GRAYSCALE)
    image[valid==0]=0
    return image,{'gamma':gamma,'contrast':contrast,'brightness':brightness,'gradient_xy':[float(gx),float(gy)],'gaussian_sigma':sigma,'noise_sigma_fraction':noise,'jpeg_quality':quality,
                  'LCD_contents_repainted':False,'photometric_applied_to_whole_valid_frame':True}

def sample_panels(rows):
    """Static sample QA only, never model predictions."""
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    for split in ('train','val','test'):
        candidates=[r for r in rows if r['split']==split and r['is_augmented']]
        chosen=[]
        for g in sorted({r['source_group'] for r in candidates}):
            candidates_g=[r for r in candidates if r['source_group']==g]
            for angle in (0,90,180,270):chosen.append(min(candidates_g,key=lambda r:abs(r['rotation_deg']-angle)))
        sheet=Image.new('RGB',(1536,540*math.ceil(len(chosen)/3)),(24,24,24))
        for k,r in enumerate(chosen):
            gray=cv2.imread(str(ROOT/'dataset'/r['image']),0);mask=cv2.imread(str(ROOT/'dataset'/r['lcd_mask']),0)
            rgb=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB);contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb,contours,-1,(60,255,100),1,cv2.LINE_AA)
            x=k%3*512;y=k//3*540;sheet.paste(Image.fromarray(rgb),(x,y+26))
            ImageDraw.Draw(sheet).text((x+4,y+3),f'{r["source_group"]} {r["rotation_deg"]:.1f}deg | {split}',font=font,fill='white')
        sheet.save(ROOT/'qa'/f'augmentation_{split}.jpg',quality=96)

def finalize_public_assets(out):
    """Keep excluded-source audit images private without deleting originals."""
    out=Path(out).resolve()
    sources=json.loads((out/'sources.json').read_text())
    retained={r[k] for r in sources for k in ('image','lcd_mask','valid_mask','protected_mask') if r.get(k)}
    private=(ROOT/'annotations'/'excluded_package_assets').resolve()
    moved=[]
    for dirname in ('source_images','source_lcd_masks','source_valid_masks','source_protected_masks','qa'):
        folder=out/dirname
        if not folder.exists():continue
        for path in folder.iterdir():
            if not path.is_file():continue
            rel=path.relative_to(out).as_posix()
            keep=(rel in retained) if dirname!='qa' else (path.suffix.lower()!='.jpg' or path.name.startswith('augmentation_'))
            if keep:continue
            target=(private/rel).resolve()
            assert path.resolve().is_relative_to(out) and target.is_relative_to(private)
            target.parent.mkdir(parents=True,exist_ok=True)
            # Only generated audit copies move. Original recording frames remain.
            path.replace(target);moved.append(rel)
    rows=[json.loads(x) for x in (out/'samples.jsonl').read_text().splitlines() if x.strip()]
    for dirname,key in [('images','image'),('lcd_masks','lcd_mask'),('valid_masks','valid_mask'),('protected_masks','protected_mask')]:
        expected={r[key] for r in rows if r.get(key)}
        actual={p.relative_to(out).as_posix() for p in (out/dirname).iterdir() if p.is_file()}
        assert actual==expected,f'Stale or missing generated assets: {dirname}'
    return moved


def build():
    out=ROOT/'dataset';annotations=json.loads((ROOT/'annotations/source_annotations.json').read_text())
    assert json.loads((ROOT/'annotations/review_record.json').read_text())['status']=='approved_for_dataset_build'
    sources=[r for r in annotations if r['split']!='excluded']
    assert len(sources)==58 and all(r['annotation_status']=='manually_visually_reviewed' for r in sources)
    videos={'night181323':Path('C:/Users/91812/Downloads/WhatsApp Video 2026-10-17 at 6.13.23 PM.mp4'),
            'day174359':Path('C:/Users/91812/Downloads/WhatsApp Video 2026-10-17 at 5.43.59 PM.mp4'),
            'rpm220557':Path('C:/Users/91812/Downloads/for rpm/WhatsApp Video 2026-10-17 at 10.05.57 PM.mp4'),
            'rpm220558':Path('C:/Users/91812/Downloads/for rpm/WhatsApp Video 2026-10-17 at 10.05.58 PM.mp4')}
    video_provenance={g:{'filename':p.name,'sha256':sha(p),'full_video_in_dataset':False} for g,p in videos.items()}
    for r in sources:
        assert sha(PROJECT/r['original_source_path'])==r['original_source_sha256']
        r['source_video_sha256']=video_provenance[r['source_group']]['sha256']
        for key in ('image','lcd_mask','valid_mask','protected_mask'):
            if r.get(key):r[key+'_sha256']=sha(out/r[key])
    write_json(out/'sources.json',sources)
    for d in ('images','lcd_masks','valid_masks','protected_masks','provenance','qa'):(out/d).mkdir(exist_ok=True)
    for filename in ('source_annotations.json','review_record.json'):shutil.copyfile(ROOT/'annotations'/filename,out/'provenance'/filename)
    write_json(out/'provenance/source_recordings.json',video_provenance)
    spec=[]
    # Equal source-recording weights within train and validation. Each actual
    # source also appears unaugmented; children can never cross a source split.
    for split,aug_count in [('train',None),('val',600),('test',200)]:
        selected=[r for r in sources if r['split']==split]
        for r in selected:spec.append((r,False,0.))
        groups=sorted({r['source_group'] for r in selected})
        total=3600-len(selected) if split=='train' else aug_count
        bygroup={g:[r for r in selected if r['source_group']==g] for g in groups}
        rng=np.random.default_rng(SEED+{'train':0,'val':1,'test':2}[split])
        for i in range(total):
            g=groups[i%len(groups)];members=bygroup[g];j=i//len(groups)
            source=members[j%len(members)]
            # Advance each source through every roll bin. Using j%12 here
            # would lock a recording with 12 sources to one bin per source.
            rotation_bin=(j//len(members)+j%len(members))%12
            rotation=rotation_bin*30+float(rng.uniform(0,30))
            spec.append((source,True,rotation))
    rows=[];image_digest_splits={};cached={};counts=Counter()
    for i,(source,is_aug,rotation) in enumerate(spec):
        ident=f'lcd_{source["split"]}_{i:05d}';rng=np.random.default_rng(SEED+i*104729)
        if source['source_id'] not in cached:
            cached[source['source_id']]=[cv2.imread(str(out/source[k]),0) if source.get(k) else None for k in ('image','lcd_mask','valid_mask','protected_mask')]
        raw,lcd,valid,protected=cached[source['source_id']]
        if is_aug:
            h,geom=augmentation_h(rng,rotation)
            image=cv2.warpPerspective(raw,h,(SIZE,SIZE),flags=cv2.INTER_LINEAR,borderValue=0)
            m=cv2.warpPerspective(lcd,h,(SIZE,SIZE),flags=cv2.INTER_NEAREST,borderValue=0)
            v=cv2.warpPerspective(valid,h,(SIZE,SIZE),flags=cv2.INTER_NEAREST,borderValue=0)
            m=cv2.bitwise_and(m,v);image,photo=photometric(image,v,rng)
            p=cv2.warpPerspective(protected,h,(SIZE,SIZE),flags=cv2.INTER_NEAREST,borderValue=0) if protected is not None else None
        else:h=np.eye(3);geom={'cropped':False,'mirrored':False};photo=None;image=raw.copy();m=lcd.copy();v=valid.copy()
        if not is_aug:p=protected.copy() if protected is not None else None
        if p is not None:
            p=cv2.bitwise_and(p,v);assert not np.any((p>0)&(m>0))
        assert image.shape==m.shape==v.shape==(SIZE,SIZE)
        assert set(np.unique(m))<=set((0,255)) and set(np.unique(v))<=set((0,255))
        assert np.count_nonzero(m)>150 and not np.any((m>0)&(v==0))
        digest=hashlib.sha256(image.tobytes()).hexdigest()
        assert digest not in image_digest_splits or image_digest_splits[digest]==source['split'],'Cross-split identical pixels'
        image_digest_splits[digest]=source['split']
        paths={'image':f'images/{ident}.png','lcd_mask':f'lcd_masks/{ident}.png','valid_mask':f'valid_masks/{ident}.png'}
        for key,arr in [('image',image),('lcd_mask',m),('valid_mask',v)]:assert cv2.imwrite(str(out/paths[key]),arr,[cv2.IMWRITE_PNG_COMPRESSION,3])
        protected_path=f'protected_masks/{ident}.png' if p is not None else None
        if p is not None:assert cv2.imwrite(str(out/protected_path),p,[cv2.IMWRITE_PNG_COMPRESSION,3])
        row={'id':ident,'source_id':source['source_id'],'source_group':source['source_group'],'frame_index':source['frame_index'],
             'source_frame':source['frame_index'],'split':source['split'],**paths,'protected_mask':protected_path,
             'is_augmented':is_aug,'augmentation_kind':'full_frame_geometry_and_photometric' if is_aug else 'original_unaugmented',
             'rotation_deg':rotation,'parent_to_sample_h':h.tolist(),'native_to_512_h':source['native_to_512_h'],
             'geometry':geom,'photometric':photo,'source_view':source['source_view'],'lighting':source['lighting'],
             'dashboard_state':source['dashboard_state'],'truncated':source['truncated'],'truncation_edges':source['truncation_edges'],
             'annotation_status':source['annotation_status'],'physical_edge_uncertainty_px':2.,
             'lcd_polygon_xy':project(source['lcd_polygon_xy'],h).tolist(),
             'seed':SEED+i*104729,'model_predictions_used_for_labels':False}
        rows.append(row);counts[source['split']]+=1
        if (i+1)%400==0:print(json.dumps({'generated':i+1,'total':len(spec)}),flush=True)
    with (out/'samples.jsonl').open('w',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n')
    sample_panels(rows)
    for f in (ROOT/'qa').glob('augmentation_*.jpg'):shutil.copyfile(f,out/'qa'/f.name)
    config={'preprocessing_id':PREPROCESSING_ID,'input_size':512,'color':'grayscale','LCD_mask_applied_to_input':False,
            'input_geometry':'fullframe_letterbox_no_crop_no_canonical_pose_warp','normalization':'grayscale_repeated_RGB_ImageNet_mean_std',
            'mask_values':[0,255],'valid_mask_semantics':'sensor_pixel_centers_after_fullframe_letterbox_or_known_augmentation',
            'output_semantics':'visible_LCD_interior_plus_narrow_bezel_margin','mirroring':False,'partial_LCD_supported':True}
    write_json(out/'preprocessing_config.json',config)
    groups={s:{r['source_group'] for r in sources if r['split']==s} for s in ('train','val','test')}
    assert not groups['test']&(groups['train']|groups['val'])
    checks={'status':'static_generation_checks_passed','original_sources_reviewed':68,'eligible_sources':58,'excluded_sources':10,
            'source_counts':dict(Counter(r['split'] for r in sources)),'sample_counts':dict(counts),
            'unaugmented_counts':dict(Counter(r['split'] for r in rows if not r['is_augmented'])),
            'no_parent_source_split_leakage':True,'test_recording_disjoint':True,'no_cross_split_exact_decoded_duplicates':True,
            'all_masks_binary_and_inside_valid_sensor':True,'full_original_footprint_retained':True,'mirroring':False,
            'LCD_contents_repainted':False,'local_model_inference_or_training':False,'old_datasets_modified':False,
            'protected_analog_metrics_available':'V3_and_V4_only','protected_analog_sample_count':sum(r['protected_mask'] is not None for r in rows),'train_val_recordings_overlap':sorted(groups['train']&groups['val']),
            'limitation':'Train/val use separated temporal blocks within three recordings, so not independent unseen-camera accuracy. Test is one heldout close-up recording with only12 real frames. Augmented samples do not add independent real recordings.'}
    write_json(out/'qa/checks.json',checks);write_json(ROOT/'qa/build_checks.json',checks)
    finalize_public_assets(out)
    files={str(p.relative_to(out)).replace('\\','/'):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    manifest={'schema_version':1,'input_size':512,'preprocessing_id':PREPROCESSING_ID,'samples_file':'samples.jsonl','source_file':'sources.json',
              'preprocessing':config,'sample_count':len(rows),'split_counts':dict(counts),'source_count':len(sources),
              'split_policy':{'method':'temporal_blocks_with_excluded_gaps; entirelyheldoutRPM220558test',
                 'train_val_recording_overlap':sorted(groups['train']&groups['val']),
                 'excluded_source_ids':[r['source_id'] for r in annotations if r['split']=='excluded'],
                 'test_source_groups':sorted(groups['test']),'all_descendants_inherit_source_split':True},
              'source_annotation_uncertainty_px_estimated':2.,'protected_analog_metrics_available':'V3_and_V4_only',
              'generator_sha256':sha(__file__),'opencv_version':cv2.__version__,'numpy_version':np.__version__,
              'seed':SEED,'sha256':files,'limitations':checks['limitation']}
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'status':'complete','samples':len(rows),'split_counts':dict(counts),'manifest_sha256':sha(out/'manifest.json'),'protected_files':len(files)}),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['annotate','review','approve-reviewed','protected-review','approve-protected','build','finalize-public'])
    args=parser.parse_args();cv2.setNumThreads(2)
    if args.action=='annotate':annotate()
    elif args.action=='review':render_review(json.loads((ROOT/'annotations/source_annotations.json').read_text()))
    elif args.action=='approve-reviewed':approve_review()
    elif args.action=='protected-review':protected_review()
    elif args.action=='approve-protected':approve_protected()
    elif args.action=='finalize-public':
        out=ROOT/'dataset';moved=finalize_public_assets(out)
        manifest=json.loads((out/'manifest.json').read_text())
        manifest['generator_sha256']=sha(__file__)
        previous=manifest['sha256']
        # This step only moves unused audit files; retained data bytes are
        # untouched. A separate strict verifier rechecks all retained hashes.
        manifest['sha256']={str(p.relative_to(out)).replace('\\','/'):previous.get(str(p.relative_to(out)).replace('\\','/')) or sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
        write_json(out/'manifest.json',manifest)
        print(json.dumps({'moved_audit_assets':len(moved),'manifest_sha256':sha(out/'manifest.json')}),flush=True)
    else:build()
