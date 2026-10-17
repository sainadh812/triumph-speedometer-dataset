"""Extract private day-video development sources and review transferred labels."""
from pathlib import Path
import json, hashlib
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
V3=ROOT.parent/'real_adaptation_v3'
VIDEO=Path(r'C:\Users\91812\Downloads\WhatsApp Video 2026-10-17 at 5.43.59 PM.mp4')
FRAMES=list(range(0,600,30))

def project(p,h):
    p=np.c_[np.asarray(p),np.ones(len(p))]@np.asarray(h).T
    return p[:,:2]/p[:,2:]

def prepare(frame):
    h,w=frame.shape[:2]; side=max(h,w); top=(side-h)//2; left=(side-w)//2
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.copyMakeBorder(gray,top,side-h-top,left,side-w-left,cv2.BORDER_CONSTANT,value=0)
    return cv2.resize(gray,(512,512),interpolation=cv2.INTER_LINEAR)

def main():
    cv2.setNumThreads(2)
    for name in ['sources/real_frames','sources/aligned','qa']:(ROOT/name).mkdir(parents=True,exist_ok=True)
    cal=json.loads((V3/'dataset500/calibration.json').read_text())
    rows=[json.loads(s) for s in (V3/'video_inference_174359_epoch8_unmasked/predictions.jsonl').read_text().splitlines()]
    refpoints=[cal['pivot_xy']]+[t['xy'] for t in cal['ticks']]
    cap=cv2.VideoCapture(str(VIDEO)); meta=[]
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',14)
    sheets=[Image.new('RGB',(1280,1340),(24,24,24)) for _ in range(2)]
    for idx in range(600):
        ok,raw=cap.read(); assert ok
        if idx not in FRAMES:continue
        gray=prepare(raw); name=f'frame_{idx:04d}.png'; cv2.imwrite(str(ROOT/'sources/real_frames'/name),gray)
        reg=rows[idx]['image_registration']; H=np.array(reg['image_to_reference_h']); inverse=np.linalg.inv(H)
        points=project(refpoints,inverse)
        cv2.imwrite(str(ROOT/'sources/aligned'/name),cv2.warpPerspective(gray,H,(512,512)))
        overlay=Image.fromarray(gray).convert('RGB');d=ImageDraw.Draw(overlay)
        for i,(x,y) in enumerate(points):
            color='#fb5ecf' if i==0 else '#31eeef';d.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
            d.text((x+4,y-8),'P' if i==0 else str(cal['ticks'][i-1]['speed_kmh']),font=font,fill=color,stroke_width=1,stroke_fill='black')
        d.text((100,5),f'Frame {idx} | {reg["inliers"]} inliers',font=font,fill='white')
        overlay.save(ROOT/'qa'/f'transferred_{idx:04d}.png')
        j=FRAMES.index(idx); sheet=sheets[j//12];sheet.paste(overlay.resize((320,320)),((j%4)*320,((j%12)//4)*335))
        meta.append({'source_frame':idx,'image':f'sources/real_frames/{name}','image_to_reference_h':H.tolist(),'registration':reg,'pivot_and_ticks_xy':points.tolist(),'label_status':'transferred_manual_draft_pending_visual_review'})
    cap.release()
    for i,sheet in enumerate(sheets):sheet.save(ROOT/'qa'/f'transferred_contact_{i+1}.jpg',quality=95)
    data={'video_name':VIDEO.name,'video_sha256':hashlib.sha256(VIDEO.read_bytes()).hexdigest(),'sources':meta,'calibration_source':'real_adaptation_v3/dataset500/calibration.json','labels_from_model_predictions':False}
    (ROOT/'sources/extraction.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'frames':len(meta),'output':str(ROOT/'qa/transferred_contact_1.jpg')}))

if __name__=='__main__':main()
