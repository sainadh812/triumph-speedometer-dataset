"""Affordable deterministic masked day-view speedometer data with 12 keypoints.

Run --preview for 20 review images; --generate only after reviewing plates/labels.
No generated values are claims of actual road speed or independent real accuracy.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, shutil, zipfile
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from prepare_sources import ROOT, project

SEED=2026101704
POLYGON=[[300,161],[417,123],[417,248],[361,265],[350,260],[300,180]]
KEYPOINTS=['pivot','needle_tip']+[f'tick_{v}' for v in range(0,181,20)]
SIZE=512

def dump(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def render_needle(plate,pivot,ticks,speed,rng):
    angles=np.unwrap(np.arctan2(ticks[:,1]-pivot[1],ticks[:,0]-pivot[0]));radii=np.linalg.norm(ticks-pivot,axis=1)
    assert np.all(np.diff(angles)>0)
    theta=np.interp(speed,np.arange(0,181,20),angles)
    length=float(np.interp(speed,np.arange(0,181,20),radii)+3.)
    direction=np.array([np.cos(theta),np.sin(theta)]);perp=np.array([-direction[1],direction[0]])
    tip=pivot+direction*length;tail=pivot-direction*24
    width=float(rng.uniform(2.5,3.3));intensity=float(rng.uniform(240,255))
    points=np.array([tail+perp*width, pivot+direction*22+perp*width*.75,tip+perp*.25,tip-perp*.25,pivot+direction*22-perp*width*.75,tail-perp*width])
    high=np.zeros((1024,1024),np.uint8);cv2.fillPoly(high,[np.rint(points*2).astype(np.int32)],255,cv2.LINE_AA)
    alpha=cv2.resize(high,(512,512),interpolation=cv2.INTER_AREA).astype(float)/255
    result=np.rint(plate*(1-alpha)+intensity*alpha).clip(0,255).astype(np.uint8)
    return result,tip,alpha,{'angle_reference_radians':float(theta),'length_reference_px':length,'half_width_px':width,'gray_level':intensity,'tail_length_px':24.}

def geometric(rng,pivot,points,enabled=True):
    if not enabled:return np.eye(3),{'roll_degrees':0,'translation_xy':[0,0],'scale':1,'shear_xy':[0,0],'perspective_corner_offsets':[[0,0]]*4}
    for attempt in range(1000):
        angle=float(rng.uniform(-30,30));scale=float(rng.uniform(.85,1.1));delta=rng.uniform(-24,24,2);shear=rng.uniform(-.08,.08,2)
        affine=np.eye(3);affine[:2]=cv2.getRotationMatrix2D(tuple(pivot),angle,scale);affine[:2,2]+=delta
        sh=np.array([[1,shear[0],-shear[0]*pivot[1]],[shear[1],1,-shear[1]*pivot[0]],[0,0,1.]])
        corners=np.array([[0,0],[511,0],[511,511],[0,511]],np.float32);offset=rng.uniform(-7,7,(4,2)).astype(np.float32)
        h=cv2.getPerspectiveTransform(corners,corners+offset)@sh@affine
        transformed=project(points,h)
        if np.all((transformed>=8)&(transformed<=503)):
            return h,{'roll_degrees':angle,'translation_xy':delta.tolist(),'scale':scale,'shear_xy':shear.tolist(),'perspective_corner_offsets':offset.tolist()}
    raise RuntimeError('Could not sample in-frame geometry')

def appearance(image,rng,profile):
    x=image.astype(np.float32)/255
    meta={'profile':profile,'gain':1.,'gamma':1.,'noise_std_255':0.,'gaussian_sigma':0.,'motion_length':0,'jpeg_quality':None}
    if profile=='clean':return image,meta
    meta['gain']=float(rng.uniform(.72,1.18));meta['gamma']=float(rng.uniform(.85,1.25));x=meta['gain']*np.maximum(x,0)**meta['gamma']
    yy,xx=np.mgrid[:512,:512]
    if profile=='shadow':
        offset=float(rng.uniform(180,320));slope=float(rng.uniform(-.6,.6));strength=float(rng.uniform(.15,.35))
        x*=1-strength*(1+np.tanh((xx+slope*yy-offset)/75))/2
        meta['shadow']={'offset':offset,'slope':slope,'strength':strength}
    if profile=='reflection':
        center=rng.uniform([160,140],[330,320]);sigma=rng.uniform([45,65],[95,120]);strength=float(rng.uniform(.05,.18))
        f=np.exp(-.5*(((xx-center[0])/sigma[0])**2+((yy-center[1])/sigma[1])**2))
        x+=strength*f*(1-np.clip(x,0,1));meta['reflection']={'center':center.tolist(),'sigma':sigma.tolist(),'strength':strength}
    if profile=='blur':
        meta['gaussian_sigma']=float(rng.uniform(.35,1.1));x=cv2.GaussianBlur(x,(0,0),meta['gaussian_sigma'])
    if profile=='motion':
        length=int(rng.choice([3,5]));kernel=np.zeros((length,length),np.float32)
        if rng.random()<.5:kernel[length//2,:]=1/length
        else:np.fill_diagonal(kernel,1/length)
        x=cv2.filter2D(x,-1,kernel);meta['motion_length']=length
    meta['noise_std_255']=float(rng.uniform(.5,3.5));x+=rng.normal(0,meta['noise_std_255']/255,x.shape)
    result=np.rint(np.clip(x,0,1)*255).astype(np.uint8)
    if profile=='compression':
        quality=int(rng.integers(72,96));ok,buf=cv2.imencode('.jpg',result,[cv2.IMWRITE_JPEG_QUALITY,quality]);assert ok
        result=cv2.imdecode(buf,cv2.IMREAD_GRAYSCALE);meta['jpeg_quality']=quality
    return result,meta

def overlay(image,points,speed,caption):
    result=Image.fromarray(image).convert('RGB');d=ImageDraw.Draw(result);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',12)
    d.line([tuple(points[0]),tuple(points[1])],fill='#fa5bc0',width=1)
    for j,(x,y) in enumerate(points):
        color='#fa5bc0' if j<2 else '#25e8e8';d.ellipse((x-2.5,y-2.5,x+2.5,y+2.5),outline=color,width=1)
        if j>=2:d.text((x+4,y-10),str((j-2)*20),font=font,fill=color,stroke_width=1,stroke_fill='black')
    d.rectangle((0,0,511,22),fill='#151515');d.text((6,3),f'{caption} | {speed:.1f} km/h',font=font,fill='white')
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preview',action='store_true');parser.add_argument('--generate',action='store_true');args=parser.parse_args();assert args.preview != args.generate
    cv2.setNumThreads(2);rng=np.random.default_rng(SEED)
    out=ROOT/('preview20' if args.preview else 'dataset2000')
    for name in ['images','masks','qa']:(out/name).mkdir(parents=True,exist_ok=True)
    plates=json.loads((ROOT/'sources/plates.json').read_text())['plates'];cal=json.loads((ROOT/'sources/calibration_draft.json').read_text());pivot=np.array(cal['pivot_xy']);ticks=np.array([t['xy'] for t in cal['ticks']]);cal['status']='root_visually_reviewed_day_reference_draft';cal['lcd_polygon_xy']=POLYGON
    cal['source_image']='reference.png';cal['original_private_source_image_sha256']=sha(ROOT/'sources/real_frames/frame_0000.png');cal['source_image_sha256']=sha(ROOT/'sources/reference_masked.png')
    dump(out/'calibration.json',cal);shutil.copy2(ROOT/'sources/reference_masked.png',out/'reference.png')
    if (out/'reference_unmasked.png').exists():(out/'reference_unmasked.png').unlink()
    fixed_mask=np.zeros((512,512),np.uint8);cv2.fillPoly(fixed_mask,[np.array(POLYGON,np.int32)],255)
    cv2.imwrite(str(out/'fixed_lcd_mask.png'),fixed_mask)
    ref=cv2.imread(str(out/'reference.png'),0);cv2.imwrite(str(out/'reference_masked.png'),ref)
    profiles=['clean','lighting','shadow','reflection','blur','motion','compression']
    inputs={r['source_frame']:cv2.imread(str(ROOT/r['plate']),0) for r in plates}
    if args.preview:
        speeds=[0,5,10,20,30,40,50,58,60,62,70,80,90,100,110,120,140,160,175,180]
        jobs=[(plates[i%6],speed,i%7,i>=6) for i,speed in enumerate(speeds)]
    else:
        jobs=[]
        for row in plates:
            count=400 if row['split']=='train' else 200
            speeds=(np.arange(count)+rng.random(count))/count*180;rng.shuffle(speeds);speeds[:10]=np.arange(0,181,20)
            jobs.extend((row,float(speed),int(rng.choice(np.arange(7),p=[.2,.2,.12,.12,.12,.12,.12])),True) for speed in speeds)
    rows=[];label_error=[];all_mask_clear=True
    for index,(source,speed,profile_index,augmented) in enumerate(jobs):
        image,tip,alpha,needle=render_needle(inputs[source['source_frame']],pivot,ticks,speed,rng)
        points=np.vstack([pivot,tip,ticks]);mask=fixed_mask.copy()
        shaft=np.zeros((512,512),np.uint8);cv2.line(shaft,tuple(pivot.astype(int)),tuple(np.rint(tip).astype(int)),255,5)
        assert not np.any((shaft>0)&(mask>0)),('mask covers shaft',speed)
        for point in points:
            x,y=np.rint(point).astype(int);assert mask[y,x]==0
        image[mask>0]=128
        h,geometry=geometric(rng,pivot,points,augmented)
        image=cv2.warpPerspective(image,h,(512,512),flags=cv2.INTER_LINEAR,borderValue=0)
        mask=cv2.warpPerspective(mask,h,(512,512),flags=cv2.INTER_NEAREST)
        needle_mask=cv2.warpPerspective((alpha>.3).astype(np.uint8),h,(512,512),flags=cv2.INTER_NEAREST)
        transformed=project(points,h);image,photo=appearance(image,rng,profiles[profile_index]);image[mask>0]=128
        observed=[True,True]
        for point in transformed[2:]:
            x,y=np.rint(point).astype(int);observed.append(not bool(needle_mask[max(0,y-1):y+2,max(0,x-1):x+2].any()))
        for point in transformed:
            x,y=np.rint(point).astype(int);assert mask[y,x]==0
        sid=f'v4_{index:05d}';image_path=f'images/{sid}.png';mask_path=f'masks/{sid}.png'
        assert cv2.imwrite(str(out/image_path),image);assert cv2.imwrite(str(out/mask_path),mask)
        recovered=project(transformed,np.linalg.inv(h));angles=np.unwrap(np.arctan2(ticks[:,1]-pivot[1],ticks[:,0]-pivot[0]));theta=np.arctan2(recovered[1,1]-recovered[0,1],recovered[1,0]-recovered[0,0]);theta+=round(((angles[0]+angles[-1])/2-theta)/(2*np.pi))*2*np.pi
        label_error.append(abs(float(np.interp(theta,angles,np.arange(0,181,20)))-speed))
        row={'sample_id':sid,'image_path':image_path,'mask_path':mask_path,'lcd_mask_path':mask_path,'split':source['split'],'points_xy':transformed.tolist(),'visibility':[True]*12,'observed_visibility':observed,'visibility_semantics':'amodal_label_available_and_inside_frame; observed_visibility separately estimates needle occlusion','speed_kmh':float(speed),'speed_valid':True,'source_frame':source['source_frame'],'source_group':source['source_group'],'donor_frame':source['donor_frame'],'reference_to_image_h':h.tolist(),'geometry':geometry,'appearance':photo,'needle_rendering':needle,'label_status':'synthetic_geometry_exact_relative_to_manually_reviewed_draft_calibration','image_sha256':sha(out/image_path)}
        rows.append(row)
        if args.preview or index%100==0:overlay(image,transformed,speed,f'{sid} | {source["split"]}').save(out/'qa'/f'{sid}.png')
    dump(out/'annotations.json',{'schema_version':1,'keypoint_names':KEYPOINTS,'landmark_supervision':'amodal_in_frame_landmark_availability','coordinate_system':'512x512 pixelcenters xright ydown; normalizeby511','samples':rows})
    with (out/'labels.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['sample_id','image_path','split','source_group','speed_kmh']+[f'{name}_{axis}' for name in KEYPOINTS for axis in ['x','y']])
        for r in rows:writer.writerow([r['sample_id'],r['image_path'],r['split'],r['source_group'],r['speed_kmh'],*np.array(r['points_xy']).ravel()])
    selected=rows if args.preview else [rows[i] for i in range(0,len(rows),100)]
    for labeled in [False,True]:
        sheet=Image.new('RGB',(5*256,4*282),(24,24,24));d=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
        for i,r in enumerate(selected[:20]):
            im=Image.open(out/'qa'/f'{r["sample_id"]}.png') if labeled else Image.open(out/r['image_path'])
            x=(i%5)*256;y=(i//5)*282;sheet.paste(im.resize((256,256)),(x,y));d.text((x+5,y+258),f'{r["speed_kmh"]:.1f} km/h | {r["appearance"]["profile"]}',font=font,fill='white')
        sheet.save(out/'qa'/('labels_contact.jpg' if labeled else 'inputs_contact.jpg'),quality=95)
    train_groups={r['source_group'] for r in rows if r['split']=='train'};val_groups={r['source_group'] for r in rows if r['split']=='validation'}
    train_sources={frame for r in rows if r['split']=='train' for frame in [r['source_frame'],r['donor_frame']]};val_sources={frame for r in rows if r['split']=='validation' for frame in [r['source_frame'],r['donor_frame']]}
    checks={'count':len(rows),'train':sum(r['split']=='train' for r in rows),'validation':sum(r['split']=='validation' for r in rows),'source_groups_disjoint':not bool(train_groups&val_groups),'source_donor_groups_separate':not bool(train_sources&val_sources),'unique_image_sha256':len({r['image_sha256'] for r in rows})==len(rows),'max_render_roundtrip_error_kmh':max(label_error),'all_mask_pixels128':all(np.all(cv2.imread(str(out/r['image_path']),0)[cv2.imread(str(out/r['mask_path']),0)>0]==128) for r in rows),'all_points_inside':all(np.all((np.array(r['points_xy'])>=0)&(np.array(r['points_xy'])<=511)) for r in rows),'lcd_mask_overlaps_targets_or_forward_shaft':False,'real_accuracy_claimed':False,'imagegen_calls':0,'source_recordings':1}
    assert checks['source_groups_disjoint'] and checks['max_render_roundtrip_error_kmh']<1e-6
    dump(out/'qa/checks.json',checks)
    preprocessing={'color':'grayscale','input_size':[512,512],'camera_stabilization':'image_only_day_reference_SIFT_static_scale_RANSAC','lcd_mask_applied':True,'lcd_mask_value':128,'lcd_polygon_reference_xy':POLYGON,'mask_before_geometric_augmentation':True,'fixed_mask_path':'fixed_lcd_mask.png','fixed_mask_sha256':sha(out/'fixed_lcd_mask.png'),'reference_image_path':'reference.png','reference_sha256':sha(out/'reference.png'),'canonical_calibration_sha256':sha(out/'calibration.json'),'stabilize_mask_script_sha256':sha(ROOT/'stabilize_mask.py'),'reference_footprint_x_inclusive':[95,416],'geometric_augmentations':['rotation+-30deg','translation+-24px','scale0.85-1.10','shear+-0.08','perspectivecorner+-7px'],'flipping':False}
    dump(out/'preprocessing_config.json',preprocessing)
    provenance={'method':'OpenCV deterministic compositing; realbackgrounds+localrealdonorpatch+33pxhubrepair; reconstructed taperedneedle','source_video_sha256':json.loads((ROOT/'sources/extraction.json').read_text())['video_sha256'],'source_video_published':False,'private_source_frames_published':False,'plate_sources':plates,'seed':SEED,'split_limit':'Different sourceplategroups and donors across train/validation, but all originate same shortclip. Not independent real-worldtest.','calibration_limit':'Manually reviewed draft2px approximateuncertainty; exactrenderingdoesnotestablishspeedaccuracy.'}
    dump(out/'provenance.json',provenance)
    (out/'README.md').write_text(f'''# V4 masked daytime landmark dataset

{len(rows)} synthetic grayscale images at 512 x 512 pixels. Each image labels the
needle pivot, needle tip, and ten major tick strokes from 0 to 180 km/h.

## Input preparation

The camera frame is padded to a square and resized to 512 pixels. Image-only
registration aligns the printed scale to the reviewed daytime reference. Only
after successful alignment is the fixed LCD polygon filled with gray 128.
Synthetic geometry transforms the image, binary LCD mask, and every landmark
together. Photometric augmentation is followed by restoring masked pixels to 128.
The public reference has its LCD hidden; original frames and video remain private.

## Labels

`annotations.json` contains `samples`. `points_xy` is ordered as pivot, needle tip,
then tick_0, tick_20, ..., tick_180. Coordinates are pixel centers (x right, y down);
divide by 511 for normalized coordinates. `visibility` means an amodal label is
available inside the frame. `observed_visibility` separately records whether the
rendered needle covers a tick. Speed is the requested synthetic indicated speed,
computed by interpolation between reviewed tick angles in the reference view.
It is not measured road speed.

## Sources and limitations

Six background plates come from one daytime development recording. Source plate
groups and needle-removal donor frames are separate across train and validation,
but neighbouring frames remain correlated. This is development validation, not an
independent real-world accuracy test. No ImageGen calls were made.

The old needle is removed using registered real donor pixels. A small central
33-pixel region is repaired with OpenCV, and the measured elliptical hub and tapered
needle are reconstructed. Real dial pixels outside the recorded repair areas are
preserved. Tick and pivot annotations have approximately 2-pixel visual uncertainty;
exact rendering does not establish physical speed accuracy.

`manifest.json` protects the images, masks, calibration, reference, and preprocessing
configuration with SHA-256 hashes. `qa/checks.json` records generation checks.
''',encoding='utf-8')
    files={p.relative_to(out).as_posix():sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    dump(out/'manifest.json',{'dataset':'triumph_day_masked_landmarks_v4','count':len(rows),'landmark_supervision':'amodal_in_frame_landmark_availability','preprocessing':preprocessing,'sha256':files})
    print(json.dumps({'dataset':str(out),'checks':checks}),flush=True)

if __name__=='__main__':main()
