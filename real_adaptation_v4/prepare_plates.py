"""Build day plates by local replacement with registered real donor pixels.

No generative model, inferred speed, or learned needle prediction is used.
Observed shaft angles are located by white-ridge contrast then reviewed visually.
"""
from pathlib import Path
import json, math, hashlib
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from prepare_sources import ROOT, V3, project
from stabilize_mask import FixedViewPreprocessor

PIVOT=np.array([303.,240.])
SOURCES={0:('train',480),120:('train',480),300:('train',0),480:('train',0),60:('validation',450),450:('validation',60)}
TICK_OVERRIDES={0:[355.5,355.5],20:[308.5,359.5],40:[259.5,348.0],60:[217.,322.5],80:[192.5,284.5],100:[181.,243.5],120:[186.,204.5],140:[206.,174.],160:[239.2,152.1],180:[276.9,143.2]}

def shaft_angle(image):
    angles=np.deg2rad(np.arange(90,200,.1)); radii=np.arange(32,90,.5)
    scores=[]
    for theta in angles:
        u=np.array([math.cos(theta),math.sin(theta)]);v=np.array([-u[1],u[0]])
        pts=PIVOT[None,:]+radii[:,None]*u
        def sample(offset):
            pos=pts+offset*v
            return cv2.remap(image,pos[:,0].astype(np.float32).reshape(1,-1),pos[:,1].astype(np.float32).reshape(1,-1),cv2.INTER_LINEAR).astype(float)[0]
        ridge=(sample(0)+sample(1)+sample(-1))/3
        background=(sample(7)+sample(-7))/2
        scores.append(np.mean(np.maximum(ridge-background,0)))
    return float(angles[int(np.argmax(scores))]),float(max(scores))

def main():
    cv2.setNumThreads(2)
    ext=json.loads((ROOT/'sources/extraction.json').read_text()); rows={r['source_frame']:r for r in ext['sources']}
    H0=np.array(rows[0]['image_to_reference_h']); inv0=np.linalg.inv(H0)
    cal=json.loads((V3/'dataset500/calibration.json').read_text());ticks=project([r['xy'] for r in cal['ticks']],inv0)
    for index,tick in enumerate(cal['ticks']):
        if tick['speed_kmh'] in TICK_OVERRIDES:ticks[index]=TICK_OVERRIDES[tick['speed_kmh']]
    cal.update(pivot_xy=PIVOT.tolist(),ticks=[{**t,'xy':xy.tolist(),'quality':'visually_reviewed_transfer_from_real_reference'} for t,xy in zip(cal['ticks'],ticks)],source_image='sources/real_frames/frame_0000.png',method='Day frame0 canonical. Tick candidates transferred using image-only SIFT, then all10 major-stroke midpoints independently visually reviewed and refined on actual daytimeimage. Hub center independently visually annotated, not copied from V3 or model output.',status='visually_reviewed_draft_with_annotation_uncertainty',notes=['Manual draft labels; tick visual uncertainty about2px andpivot about2px, not measuredconfidence.','Allsourcesfromsame development clip; syntheticvalidation isnotindependentrealtest.'])
    cal.pop('lcd_polygon_xy',None)
    cal['source_image_sha256']=hashlib.sha256((ROOT/'sources/real_frames/frame_0000.png').read_bytes()).hexdigest()
    for tick in cal['ticks']:
        tick['notes']='Actual daytime major-stroke midpoint between two printedrails, visually reviewed at enlarged scale; approximately2px annotation uncertainty.'+(' Zero endpoint is visible in this day frame.' if tick['speed_kmh']==0 else '')
        tick['quality']='observed_daytime_visual_draft' if tick['speed_kmh'] in TICK_OVERRIDES else 'visually_reviewed_real_image_transfer'
    (ROOT/'sources/calibration_draft.json').write_text(json.dumps(cal,indent=2)+'\n')
    aligned={};registrations={}
    mask_polygon=[[300,161],[417,123],[417,248],[361,265],[350,260],[300,180]]
    public_reference=cv2.imread(str(ROOT/rows[0]['image']),0)
    reference_mask=np.zeros((512,512),np.uint8);cv2.fillPoly(reference_mask,[np.array(mask_polygon,np.int32)],255)
    public_reference[reference_mask>0]=128;public_reference[:,:95]=0;public_reference[:,417:]=0
    cv2.imwrite(str(ROOT/'sources/reference_masked.png'),public_reference)
    processor=FixedViewPreprocessor(public_reference,mask_polygon,ticks,PIVOT)
    for idx in SOURCES:
        image=cv2.imread(str(ROOT/rows[idx]['image']),0)
        processed,registration=processor.process(image,apply_mask=False)
        assert registration['accepted'], (idx,registration)
        h=np.array(registration['image_to_reference_h']);registrations[idx]=registration
        aligned[idx]=processed
        cv2.imwrite(str(ROOT/'sources'/f'day_aligned_{idx:04d}.png'),aligned[idx])
    records=[]; font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    sheet=Image.new('RGB',(1536,1100),(24,24,24));d=ImageDraw.Draw(sheet)
    yy,xx=np.mgrid[:512,:512]; radius=np.hypot(xx-PIVOT[0],yy-PIVOT[1])
    for k,(idx,(split,donor_idx)) in enumerate(SOURCES.items()):
        original=aligned[idx]; donor=aligned[donor_idx];theta,score=shaft_angle(original);u=np.array([math.cos(theta),math.sin(theta)])
        old_tip=PIVOT+u*125
        footprint=np.zeros((512,512),np.uint8)
        cv2.line(footprint,tuple(np.rint(PIVOT-u*32).astype(int)),tuple(np.rint(old_tip).astype(int)),255,12,cv2.LINE_AA)
        footprint[radius<18]=0
        ring=cv2.dilate(footprint,np.ones((17,17),np.uint8))-cv2.dilate(footprint,np.ones((7,7),np.uint8))
        valid=(ring>0)&(radius>25)&(radius<120)
        offset=float(np.median(original[valid].astype(float)-donor[valid].astype(float)))
        fixed=np.clip(donor.astype(float)+offset,0,255)
        alpha=cv2.GaussianBlur(footprint,(5,5),.7).astype(float)/255
        # Preserve actual source texture outside only the previous needle footprint.
        clean=np.rint(original*(1-alpha)+fixed*alpha).clip(0,255).astype(np.uint8)
        # Donor shafts/countertails all cross near the hub. Inpaint the small
        # central disk BEFORE putting the observed reference cap pixels back;
        # otherwise white cap pixels bleed into a repaired tail as a second needle.
        center_repair=(radius<33).astype(np.uint8)*255
        clean=cv2.inpaint(clean,center_repair,5,cv2.INPAINT_TELEA)
        cap_mask=np.zeros((512,512),np.uint8)
        cv2.ellipse(cap_mask,tuple(PIVOT.astype(int)),(20,19),-25,0,360,255,-1,cv2.LINE_AA)
        cap_alpha=cv2.GaussianBlur(cap_mask,(3,3),.55).astype(float)/255
        cap_level=float(np.median(original[radius<12]))
        cap_texture=np.clip(cap_level+.08*(xx-PIVOT[0])+.04*(yy-PIVOT[1]),0,255)
        clean=np.rint(clean*(1-cap_alpha)+cap_texture*cap_alpha).clip(0,255).astype(np.uint8)
        alpha=np.maximum(alpha,center_repair.astype(float)/255)
        cv2.imwrite(str(ROOT/'sources'/f'plate_{idx:04d}.png'),clean)
        cv2.imwrite(str(ROOT/'sources'/f'repair_mask_{idx:04d}.png'),np.rint(alpha*255).astype(np.uint8))
        overlay=Image.fromarray(original).convert('RGB');od=ImageDraw.Draw(overlay)
        od.line([tuple(PIVOT-u*32),tuple(old_tip)],fill='#ff33bf',width=1)
        for t,p in zip(cal['ticks'],ticks):
            x,y=p;od.ellipse((x-2,y-2,x+2,y+2),outline='#22eaff',width=1)
        x,y=PIVOT;od.ellipse((x-3,y-3,x+3,y+3),outline='#ff33bf',width=2)
        row=k//3;col=k%3
        sheet.paste(overlay,(col*512,row*550+25));sheet.paste(Image.fromarray(clean).resize((256,256)),(col*512+256,row*550+275))
        d.text((col*512+5,row*550+2),f'{idx} {split} | donor {donor_idx} | plate inset',fill='white',font=font)
        records.append({'source_frame':idx,'source_group':f'day174359_plate_{idx:04d}','split':split,'donor_frame':donor_idx,'plate':f'sources/plate_{idx:04d}.png','source_to_day_reference_h':registrations[idx]['image_to_reference_h'],'registration':registrations[idx],'repair_mask':f'sources/repair_mask_{idx:04d}.png','observed_shaft_angle_degrees':math.degrees(theta),'ridge_score':score,'needle_removal_method':'registered real donor shaft pixels; central33px disk OpenCV Telea repair with measured ellipse cap reconstructed at locally measuredbrightness','donor_brightness_offset':offset,'cap_gray_level':cap_level,'outside_repair_mask_unchanged':bool(np.array_equal(clean[alpha==0],original[alpha==0]))})
    (ROOT/'sources/plates.json').write_text(json.dumps({'status':'root_visually_reviewed_rendering_and_needle_removal','pivot_xy':PIVOT.tolist(),'plates':records,'imagegen_calls':0},indent=2)+'\n')
    sheet.save(ROOT/'qa/plate_review.jpg',quality=95)
    print(json.dumps(records))

if __name__=='__main__':main()
