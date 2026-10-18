"""Pure CPU dataset checks and visual annotation sheets; no model execution."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
try:
    from .dataset import validate_dataset
except ImportError:
    from dataset import validate_dataset

def render(dataset_root,output_root):
    data=validate_dataset(dataset_root);root=Path(dataset_root);out=Path(output_root);out.mkdir(parents=True,exist_ok=True)
    report={'status':'numeric_validation_passed','manifest_sha256':data['manifest_sha256'],'tasks':{}}
    for task,info in data['tasks'].items():
        rows=info['rows'];chosen=[r for r in rows if r['id'].endswith('_000')]
        # Also inspect rotation/mirror/appearance examples, not just clean images.
        augmented=[r for r in rows if r['split']=='train' and r['id'].endswith(('_008','_010','_011','_023'))]
        chosen+=augmented[::max(1,len(augmented)//12)]
        taskout=out/task;taskout.mkdir(exist_ok=True)
        cards=[]
        for r in chosen:
            img=cv2.imread(str(root/task/r['image'])); h,w=img.shape[:2]
            if task=='localizer':
                mask=cv2.imread(str(root/task/r['lcd_mask']),0)
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(img,contours,-1,(255,180,0),1)
                if r['present']:
                    box=np.round(r['dashboard_box']).astype(int)
                    cv2.rectangle(img,tuple(box[:2]),tuple(box[2:]),(0,180,255),1)
            for index,(point,visible) in enumerate(zip(r['points'],r['visible'])):
                if not visible:continue
                p=tuple(np.round(point).astype(int))
                color=(30,220,70) if task!='rpm' or index<7 else (20,30,255)
                cv2.circle(img,p,3,color,-1)
                name=info['point_names'][index].replace('tick_','').replace('rpm_','').replace('lcd_','')
                origin=(min(w-35,p[0]+4),max(12,p[1]-3))
                cv2.putText(img,name,origin,cv2.FONT_HERSHEY_SIMPLEX,.35,(0,0,0),2,cv2.LINE_AA)
                cv2.putText(img,name,origin,cv2.FONT_HERSHEY_SIMPLEX,.35,color,1,cv2.LINE_AA)
            card=np.full((h+44,w,3),245,np.uint8);card[44:]=img
            cv2.putText(card,r['id'],(4,16),cv2.FONT_HERSHEY_SIMPLEX,.40,(20,20,20),1,cv2.LINE_AA)
            cv2.putText(card,r['split']+' | '+str(sum(r['visible']))+' visible',(4,35),cv2.FONT_HERSHEY_SIMPLEX,.38,(20,20,20),1,cv2.LINE_AA)
            cv2.imwrite(str(taskout/(r['id']+'.jpg')),card,[cv2.IMWRITE_JPEG_QUALITY,95])
            cards.append(card)
        for page in range(0,len(cards),12):
            subset=cards[page:page+12];cw,ch=subset[0].shape[1],subset[0].shape[0]
            scale=min(1,330/cw);thumbw,thumbh=int(cw*scale),int(ch*scale)
            canvas=np.full((thumbh*((len(subset)+3)//4),thumbw*4,3),230,np.uint8)
            for n,card in enumerate(subset):
                y=(n//4)*thumbh;x=(n%4)*thumbw
                canvas[y:y+thumbh,x:x+thumbw]=cv2.resize(card,(thumbw,thumbh))
            cv2.imwrite(str(out/f'{task}_sheet_{page//12:02d}.jpg'),canvas,[cv2.IMWRITE_JPEG_QUALITY,94])
        report['tasks'][task]={'counts':info['counts'],'rendered_review_images':len(chosen),
                              'visible_point_counts':dict(zip(info['point_names'],np.array([r['visible'] for r in rows]).sum(0).tolist()))}
    (out/'numeric_audit.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    render(a.dataset,a.output)
