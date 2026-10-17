"""Generate 20 deterministic, review-only real-background speedometer examples.

No model, training or uploads. Run after prepare_frames/register_frames and
reviewed draft annotations + prepare_plates. Labels describe rendering geometry;
physical dial agreement is limited by the manually estimated calibration.
"""
from pathlib import Path
import csv,hashlib,json,math,zipfile
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from register_frames import project

ROOT=Path(__file__).resolve().parent
SEED=2026101703
SPEEDS=[0,5,10,20,30,40,50,58,60,62,70,80,90,100,110,120,140,160,175,180]
SIZE=512

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,im):assert cv2.imwrite(str(path),im)
def xyround(xy):return tuple(np.rint(xy).astype(int))
def font(size,bold=False):return ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf' if bold else 'C:/Windows/Fonts/arial.ttf',size)
def norm(xy):return (np.asarray(xy)/511).tolist()

def angle_table(pivot,points):
    vectors=np.asarray(points)-pivot
    return np.unwrap(np.arctan2(vectors[:,1],vectors[:,0])),np.linalg.norm(vectors,axis=1)

def estimate_speed(pivot,tip,ticks,speeds):
    angles,_=angle_table(np.asarray(pivot),ticks)
    v=np.asarray(tip)-pivot;theta=math.atan2(v[1],v[0])
    theta+=round(((angles[0]+angles[-1])/2-theta)/(2*math.pi))*2*math.pi
    if theta<angles[0]-.06 or theta>angles[-1]+.06:return None
    return float(np.interp(theta,angles,speeds))

def draw_overlay(im,pivot,tip,ticks,speed,caption):
    out=Image.fromarray(im).convert('RGB');d=ImageDraw.Draw(out)
    d.line([tuple(pivot),tuple(tip)],fill='#ff65cd',width=2)
    x,y=pivot;d.ellipse((x-5,y-5,x+5,y+5),outline='#ff65cd',width=2)
    x,y=tip;d.rectangle((x-4,y-4,x+4,y+4),outline='#ff65cd',width=2)
    for v,p in ticks:
        x,y=p;d.ellipse((x-3,y-3,x+3,y+3),outline='#38daf0',width=2)
        d.text((x+5,y-9),str(v),font=font(12),fill='#38daf0',stroke_width=1,stroke_fill='black')
    d.rectangle((0,0,511,24),fill='#151515')
    d.text((7,2),caption,font=font(16,True),fill='white')
    return out

def appearance(image,rng,kind):
    x=image.astype(np.float32)/255
    meta={'profile':kind,'gain':1.,'gamma':1.,'noise_std_255':0.,'gaussian_sigma_px':0.,'motion_length_px':0,'reflection':None,'shadow':None,'jpeg_quality':None}
    if kind=='clean':return image.copy(),meta
    if kind=='dim':meta.update(gain=.66,gamma=1.16,noise_std_255=2.5)
    elif kind=='bright':meta.update(gain=1.13,gamma=.94)
    elif kind=='blur':meta.update(gaussian_sigma_px=.7,noise_std_255=1.2)
    elif kind=='motion':meta.update(motion_length_px=3,noise_std_255=1.8)
    elif kind=='compression':meta.update(jpeg_quality=74,noise_std_255=2.0)
    elif kind in ('reflection','shadow'):meta.update(gain=.98,gamma=1.02,noise_std_255=1.5)
    x=meta['gain']*np.power(x,meta['gamma'])
    yy,xx=np.mgrid[:512,:512]
    if kind=='reflection':
        c=[float(rng.uniform(120,240)),float(rng.uniform(140,260))];w=[70.,115.];strength=.13
        f=np.exp(-.5*(((xx-c[0])/w[0])**2+((yy-c[1])/w[1])**2))
        x+=strength*f*(1-np.clip(x,0,1));meta['reflection']={'center_xy':c,'sigma_xy':w,'strength':strength}
    if kind=='shadow':
        gradient=(1+np.tanh((xx+yy*.25-215)/65))/2;x*=1-.28*gradient
        meta['shadow']={'strength':.28,'softness_px':65,'offset_px':215,'slope':.25}
    if meta['gaussian_sigma_px']:x=cv2.GaussianBlur(x,(0,0),meta['gaussian_sigma_px'])
    if meta['motion_length_px']:
        kernel=np.zeros((3,3),np.float32);kernel[1,:]=1/3;x=cv2.filter2D(x,-1,kernel)
    if meta['noise_std_255']:x+=rng.normal(0,meta['noise_std_255']/255,x.shape)
    result=np.rint(np.clip(x,0,1)*255).astype(np.uint8)
    if meta['jpeg_quality']:
        ok,buf=cv2.imencode('.jpg',result,[cv2.IMWRITE_JPEG_QUALITY,meta['jpeg_quality']]);assert ok
        result=cv2.imdecode(buf,cv2.IMREAD_GRAYSCALE)
    return result,meta

def make_real_labels(calibration,registrations):
    out=ROOT/'real_review';out.mkdir(exist_ok=True)
    for name in ['images','overlays','lcd_masks']:(out/name).mkdir(exist_ok=True)
    annotations=[r for file in ['needle_a.json','needle_b.json'] for r in json.loads((ROOT/'annotations'/file).read_text())]
    manifest=json.loads((ROOT/'real_frames_manifest.json').read_text())
    by_frame={r['frame_index']:r for r in manifest['frames']}
    ref_ticks=np.array([r['xy'] for r in calibration['ticks']]);speeds=[r['speed_kmh'] for r in calibration['ticks']]
    lcd=np.array(calibration['lcd_polygon_xy']);rows=[]
    for ann in annotations:
        index=ann['frame_index'];reg=registrations[index]
        h=np.array(reg['source_to_reference_h']);inverse=np.linalg.inv(h)
        ticks=project(ref_ticks,inverse);polygon=project(lcd,inverse)
        im=cv2.imread(str(ROOT/by_frame[index]['file']),0)
        mask=np.zeros_like(im);cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],255)
        mask=cv2.dilate(mask,np.ones((5,5),np.uint8));im[mask>0]=128
        speed=estimate_speed(ann['pivot_xy'],ann['tip_xy'],ticks,speeds) if ann['confidence']!='low' else None
        row={**ann,'image':f'images/frame_{index:04d}.png','lcd_mask':f'lcd_masks/frame_{index:04d}.png','split':'development_review','label_status':'manual_visual_draft_pending_user_review','coordinate_system':'512x512 pixel centers; x right y down','needle_xy_normalized':{'pivot':norm(ann['pivot_xy']),'tip':norm(ann['tip_xy'])},'speed_kmh_calibrated_estimate':speed,'speed_label_kind':'approximate_indicated_needle_speed_not_road_speed','use_for_training_by_default':False,'eligible_after_user_review':ann['confidence']!='low','registration_status':reg['registration_status'],'tick_labels_kind':'transferred_from_manually_annotated_real_reference_via_estimated_registration','tick_landmarks':[{'speed_kmh':v,'xy':p.tolist(),'quality':'transferred_draft_not_independently_verified_in_this_frame'} for v,p in zip(speeds,ticks)],'lcd_polygon_xy':polygon.tolist()}
        save(out/row['image'],im);save(out/row['lcd_mask'],mask)
        text=f"Frame {index} | manual draft | {'REVIEW' if speed is None else f'~{speed:.1f} km/h'}"
        draw_overlay(im,ann['pivot_xy'],ann['tip_xy'],zip(speeds,ticks),speed,text).save(out/'overlays'/f'frame_{index:04d}.png')
        rows.append(row)
    dump(out/'labels.json',{'status':'development_review_only','frames':rows})
    for part in range(2):
        sheet=Image.new('RGB',(4*256,3*280),(25,25,25));d=ImageDraw.Draw(sheet)
        for j,row in enumerate(rows[part*12:(part+1)*12]):
            tile=Image.open(out/'overlays'/f"frame_{row['frame_index']:04d}.png").resize((256,256))
            x=j%4*256;y=j//4*280;sheet.paste(tile,(x,y));d.text((x+5,y+257),f"Frame {row['frame_index']} | {row['confidence']}",fill='white',font=font(15))
        sheet.save(ROOT/'qa'/f'real_labels_{part+1}.jpg',quality=95)
    return rows

def main():
    rng=np.random.default_rng(SEED)
    cal=json.loads((ROOT/'annotations/reference_ticks.json').read_text())
    prov=json.loads((ROOT/'sources/provenance.json').read_text())
    regs={r['frame_index']:r for r in json.loads((ROOT/'registration.json').read_text())['frames']}
    real_rows=make_real_labels(cal,regs)
    out=ROOT/'preview20'
    for name in ['images','overlays','needle_masks','lcd_masks']:(out/name).mkdir(parents=True,exist_ok=True)
    ref_pivot=np.array(cal['pivot_xy']);ref_ticks=np.array([t['xy'] for t in cal['ticks']]);speeds=[t['speed_kmh'] for t in cal['ticks']]
    raw_alpha=np.load(ROOT/'sources/needle_alpha.npy')
    plate_rows={p['frame_index']:p for p in prov['plates']}
    profiles=['clean','dim','reflection','clean','blur','clean','shadow','clean','clean','clean','motion','bright','compression','reflection','dim','clean','shadow','blur','bright','clean']
    sources=[0,180,204,180,204,0,180,0,0,0,204,180,204,0,180,0,204,180,204,0]
    rows=[];errors=[]
    for i,(speed,profile,source) in enumerate(zip(SPEEDS,profiles,sources),1):
        plate_info=plate_rows[source];pivot=np.array(plate_info['pivot_xy'])
        angles,radii=angle_table(pivot,ref_ticks)
        assert np.all(np.diff(angles)>0)
        theta=float(np.interp(speed,speeds,angles))
        radius=float(np.interp(speed,speeds,radii)+4.)
        target_tip=pivot+radius*np.array([math.cos(theta),math.sin(theta)])
        # Actual source-needle alpha is transformed with exactly the same endpoint geometry.
        turn=theta-math.pi/2;scale=radius/115.;c=math.cos(turn)*scale;s=math.sin(turn)*scale
        needle_h=np.array([[c,-s,0],[s,c,0],[0,0,1.]],float)
        needle_h[:2,2]=pivot-(needle_h[:2,:2]@ref_pivot)
        alpha=cv2.warpPerspective(raw_alpha,needle_h,(512,512),flags=cv2.INTER_LINEAR)
        alpha=np.clip(alpha*(1.2 if source else 1),0,1)
        image=cv2.imread(str(ROOT/plate_info['plate']),0).astype(float)
        image=np.rint(image*(1-alpha)+255*alpha).clip(0,255).astype(np.uint8)
        lcd=np.zeros((512,512),np.uint8);cv2.fillPoly(lcd,[np.rint(cal['lcd_polygon_xy']).astype(np.int32)],255)
        lcd=cv2.dilate(lcd,np.ones((5,5),np.uint8));image[lcd>0]=128
        camera=np.eye(3);geometry={'roll_degrees':0.,'translation_xy_px':[0.,0.],'scale':1.,'perspective_corner_offsets_px':[[0.,0.]]*4}
        # Keep calibration checks and the58/60/62comparison identical in appearance.
        if profile!='clean':
            angle=float(rng.uniform(-3,3));sc=float(rng.uniform(.985,1.015));delta=rng.uniform(-4,4,2)
            camera[:2]=cv2.getRotationMatrix2D((211,225),angle,sc);camera[:2,2]+=delta
            offsets=rng.uniform(-1.5,1.5,(4,2)).astype(np.float32);corners=np.array([[0,0],[511,0],[511,511],[0,511]],np.float32)
            camera=cv2.getPerspectiveTransform(corners,corners+offsets)@camera
            geometry={'roll_degrees':angle,'translation_xy_px':delta.tolist(),'scale':sc,'perspective_corner_offsets_px':offsets.tolist()}
        image=cv2.warpPerspective(image,camera,(512,512),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
        lcd=cv2.warpPerspective(lcd,camera,(512,512),flags=cv2.INTER_NEAREST)
        needle_mask=cv2.warpPerspective((alpha>.14).astype(np.uint8)*255,camera,(512,512),flags=cv2.INTER_NEAREST)
        needle_mask[lcd>0]=0
        image,photo=appearance(image,rng,profile);image[lcd>0]=128
        points=project([pivot,target_tip],camera);ticks=project(ref_ticks,camera)
        recovered=estimate_speed(pivot,target_tip,ref_ticks,speeds)
        errors.append(abs(recovered-speed))
        name=f'{i:02d}_{speed:03d}kmh'
        row={'id':name,'image':f'images/{name}.png','speed_kmh':speed,'speed_label_kind':'requested_synthetic_indicated_speed','label_status':'render_geometry_exact_relative_to_draft_real_tick_calibration','source_frame':source,'source_family':f'clip181323_plate_{source:04d}','split':'preview_only_no_train_validation_test_split','width':512,'height':512,'grayscale':True,'needle_position_px':{'pivot':points[0].tolist(),'tip':points[1].tolist()},'needle_position_normalized':{'pivot':norm(points[0]),'tip':norm(points[1])},'needle_angle_degrees_image_clockwise_from_right':float(math.degrees(math.atan2(points[1,1]-points[0,1],points[1,0]-points[0,0]))%360),'needle_angle_degrees_reference_unwrapped':float(math.degrees(theta)),'needle_radius_reference_px':radius,'needle_tip_radius_rule':'interpolated observed tick radius plus4px; preview approximation, not metrology','tick_landmarks':[{'speed_kmh':v,'xy':p.tolist(),'calibration_quality':t['quality']} for v,p,t in zip(speeds,ticks,cal['ticks'])],'needle_mask':f'needle_masks/{name}.png','lcd_mask':f'lcd_masks/{name}.png','lcd_gray_value':128,'reference_to_image_h':camera.tolist(),'source_needle_to_reference_h':needle_h.tolist(),'geometry':geometry,'appearance':photo,'seed':SEED,'calibration_sha256':sha(ROOT/'annotations/reference_ticks.json')}
        save(out/row['image'],image);save(out/row['needle_mask'],needle_mask);save(out/row['lcd_mask'],lcd)
        draw_overlay(image,*points,zip(speeds,ticks),speed,f'{i:02d} | {speed} km/h | {profile} | source {source}').save(out/'overlays'/f'{name}.png')
        row['image_sha256']=sha(out/row['image']);rows.append(row)
    dump(out/'labels.json',{'dataset':'real_dashboard_lcd_mask_preview20_v3','status':'user_inspection_required_before_expansion_or_training','coordinate_system':'512x512 pixel centers, x right y down; normalized by511','calibration_uncertainty':'Manual real tick/pivot drafts, about2px visually estimated; zero about3px. Rendering roundtrip does not establish real accuracy.','samples':rows})
    with (out/'labels.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['id','image','speed_kmh','pivot_x','pivot_y','tip_x','tip_y','source_frame','appearance'])
        for r in rows:writer.writerow([r['id'],r['image'],r['speed_kmh'],*r['needle_position_px']['pivot'],*r['needle_position_px']['tip'],r['source_frame'],r['appearance']['profile']])
    for labeled in [False,True]:
        sheet=Image.new('RGB',(5*256,4*283),(22,22,22));d=ImageDraw.Draw(sheet)
        for i,r in enumerate(rows):
            path=(out/'overlays'/f"{r['id']}.png") if labeled else out/r['image']
            x=i%5*256;y=i//5*283;sheet.paste(Image.open(path).resize((256,256)),(x,y))
            d.text((x+6,y+258),f"{i+1:02d} | {r['speed_kmh']} km/h | {r['appearance']['profile']}",font=font(15),fill='white')
        sheet.save(ROOT/'qa'/('synthetic20_labels.jpg' if labeled else 'synthetic20_inputs.jpg'),quality=95)
    trio=Image.new('RGB',(3*512,542),(22,22,22));d=ImageDraw.Draw(trio)
    for i,r in enumerate([r for r in rows if r['speed_kmh'] in [58,60,62]]):
        trio.paste(Image.open(out/r['image']),(i*512,30));d.text((i*512+8,4),f"{r['speed_kmh']} km/h | same source and appearance",font=font(20),fill='white')
    trio.save(ROOT/'qa/comparison_58_60_62.jpg',quality=96)
    checks={'synthetic_count':len(rows),'real_frame_count':len(real_rows),'low_confidence_real_frames':[r['frame_index'] for r in real_rows if r['confidence']=='low'],'render_label_roundtrip_max_error_kmh':max(errors),'all_lcd_pixels_exact128':all(np.all(cv2.imread(str(out/r['image']),0)[cv2.imread(str(out/r['lcd_mask']),0)>0]==128) for r in rows),'all_needle_points_inside_image':all(np.all((np.array(list(r['needle_position_px'].values()))>=0)&(np.array(list(r['needle_position_px'].values()))<512)) for r in rows),'all_images_grayscale512':all(cv2.imread(str(out/r['image']),0).shape==(512,512) for r in rows),'source_repairs_preserve_outside_mask':all(p['outside_repair_pixels_identical'] for p in prov['plates']),'imagegen_calls':1,'training_performed':False,'uploaded':False,'evaluation_accuracy_claimed':False}
    dump(ROOT/'qa/checks.json',checks)
    print(json.dumps(checks))

if __name__=='__main__':main()
