"""Register fixed-camera development frames using static dial texture only."""
from pathlib import Path
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent

def project(xy,h):
    p=np.c_[np.atleast_2d(xy),np.ones(len(np.atleast_2d(xy)))]@h.T
    return p[:,:2]/p[:,2:]

def main():
    (ROOT/'aligned').mkdir(exist_ok=True)
    rows=json.loads((ROOT/'real_frames_manifest.json').read_text())['frames']
    ref=cv2.imread(str(ROOT/rows[0]['file']),0)
    yy,xx=np.mgrid[:512,:512]; radius=np.sqrt((xx-211)**2+(yy-225)**2)
    mask=((radius>93)&(radius<188)&(xx<270)&(yy>40)&(yy<390)).astype(np.uint8)*255
    clahe=cv2.createCLAHE(2.,(8,8));sift=cv2.SIFT_create(nfeatures=3000,contrastThreshold=.012)
    kr,dr=sift.detectAndCompute(clahe.apply(ref),mask)
    results=[]
    for row in rows:
        im=cv2.imread(str(ROOT/row['file']),0)
        if row['frame_index']==0:
            h=np.eye(3);n=None;rmse=0.;quality='reference'
        elif row['frame_index']==72 and (ROOT/'registration_72_review.json').exists():
            reviewed=json.loads((ROOT/'registration_72_review.json').read_text())['selected_first_hop']
            h=np.array(reviewed['source_to_reference_h'],float)
            n=reviewed['inliers'];rmse=reviewed['rmse'];quality='repaired_low_light_review'
        else:
            # Allow shifted dial positions in the source; only target keypoints are restricted.
            k,d=sift.detectAndCompute(clahe.apply(im),None)
            pairs=cv2.BFMatcher().knnMatch(d,dr,k=2)
            good=[m for m,n in pairs if m.distance<.72*n.distance]
            a=np.float32([k[m.queryIdx].pt for m in good]);b=np.float32([kr[m.trainIdx].pt for m in good])
            h,inliers=cv2.findHomography(a,b,cv2.RANSAC,2.2,maxIters=5000)
            if h is None: raise ValueError(f'Could not register {row}')
            inside=inliers.ravel().astype(bool);n=int(inside.sum());rmse=float(np.sqrt(np.mean(np.sum((project(a[inside],h)-b[inside])**2,axis=1))))
            quality='review' if n>=12 and rmse<1.6 else 'low'
        aligned=cv2.warpPerspective(im,h,(512,512),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(str(ROOT/'aligned'/f"frame_{row['frame_index']:04d}.png"),aligned)
        results.append({'frame_index':row['frame_index'],'source_to_reference_h':h.tolist(),'static_feature_inliers':n,'inlier_rmse_px':rmse,'registration_status':quality})
    (ROOT/'registration.json').write_text(json.dumps({'reference_frame':0,'method':'SIFT/CLAHE with reference features restricted to dial annulus; RANSAC projective transform','status':'estimated_registration_for_visual_review','frames':results},indent=2)+'\n')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',18)
    for part in range(2):
        sheet=Image.new('RGB',(1280,1038),(24,24,24));draw=ImageDraw.Draw(sheet)
        for i,row in enumerate(results[part*12:(part+1)*12]):
            im=Image.open(ROOT/'aligned'/f"frame_{row['frame_index']:04d}.png").resize((320,320))
            x=i%4*320;y=i//4*346;sheet.paste(im,(x,y));draw.text((x+5,y+321),f"{row['frame_index']} | n={row['static_feature_inliers']} err={row['inlier_rmse_px']:.2f}",font=font,fill='white')
        sheet.save(ROOT/'qa'/f'aligned_{part+1}.jpg',quality=93)
    print(json.dumps(results))

if __name__=='__main__':main()
