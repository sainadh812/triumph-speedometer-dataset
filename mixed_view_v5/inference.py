"""V5 full-frame speed inference; image registration moves the LCD mask only.

No source labels, known speed, video name, frame number, camera-specific crop,
input-image homography warp, previous-frame mask, or previous speed is used.
Static public reference geometry only locates a safe LCD mask. Speed comes
from the network's twelve current-image landmarks and the frozen V4 decoder.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from real_adaptation_v4.geometry import LANDMARK_NAMES, load_calibration, read_speed

SIZE = 512
MASK_VALUE = 128
PREPROCESSING_ID = 'gray512_fullframe_native_view_lcd128_mask_v5_imagenet_normalized'
MASK_VISIBILITY_POLICY = 'projected_lcd_intersection_with_visible_sensor_v1'
MIN_VISIBLE_LCD_AREA_PX2 = 350.
PARTIAL_MASK_MIN_CONTAINMENT = .95
# Manual reference ticks are uncertain by about2–3 native512px. This fixed
# bound is declared before real-video inference and is not fitted to outputs.
PARTIAL_MASK_MAX_SEMANTIC_P90_PX = 4.


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def project(points, matrix):
    p=np.asarray(points,dtype=np.float64)
    h=np.asarray(matrix,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=2 or h.shape!=(3,3) or not np.isfinite(p).all() or not np.isfinite(h).all():
        raise ValueError('invalid_projective_inputs')
    q=np.c_[p,np.ones(len(p))]@h.T
    if np.any(np.abs(q[:,2])<1e-8) or not (np.all(q[:,2]>0) or np.all(q[:,2]<0)):
        raise ValueError('projective_horizon')
    out=q[:,:2]/q[:,2:3]
    if not np.isfinite(out).all():raise ValueError('nonfinite_projection')
    return out


def letterbox(frame):
    """Preserve all sensor pixels/orientation, pad square, then resize to512."""
    if not isinstance(frame,np.ndarray) or frame.dtype!=np.uint8 or frame.ndim not in (2,3) or min(frame.shape[:2])<2:
        raise ValueError('Expected nonempty uint8 gray/BGR image')
    if frame.ndim==3 and frame.shape[2]!=3:raise ValueError('Expected BGR channels')
    height,width=frame.shape[:2];side=max(height,width)
    left,top=(side-width)//2,(side-height)//2
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) if frame.ndim==3 else frame
    square=cv2.copyMakeBorder(gray,top,side-height-top,left,side-width-left,cv2.BORDER_CONSTANT,value=0)
    out=cv2.resize(square,(SIZE,SIZE),interpolation=cv2.INTER_LINEAR)
    scale=SIZE/side
    matrix=np.array([[scale,0,(left+.5)*scale-.5],[0,scale,(top+.5)*scale-.5],[0,0,1]],float)
    footprint=project([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],matrix)
    return out,matrix,footprint


def mask_polygon(polygon):
    polygon=np.asarray(polygon,float)
    if polygon.ndim!=2 or polygon.shape[1]!=2 or len(polygon)<3 or not np.isfinite(polygon).all():
        raise ValueError('invalid_lcd_polygon')
    mask=np.zeros((SIZE,SIZE),np.uint8)
    cv2.fillPoly(mask,[np.rint(polygon).astype(np.int32)],255)
    return mask


def mask_iou(first,second):
    a,b=first>0,second>0
    union=np.count_nonzero(a|b)
    return float(np.count_nonzero(a&b)/union) if union else 0.


def compare_reference_masks(first,first_report,second,second_report):
    """Allow a clipped LCD outline only when both dial geometries agree.

V4's reference LCD ends at its sensor edge. A fuller V3 outline can therefore
contain it without high IoU. Containment alone is insufficient: independent
reference homographies must also agree on pivot and ten semantic tick centers.
"""
    a,b=first>0,second>0
    intersection=np.count_nonzero(a&b);small=min(np.count_nonzero(a),np.count_nonzero(b))
    iou=mask_iou(first,second);containment=float(intersection/small) if small else 0.
    p=np.asarray(first_report.get('protected_points_input_xy'),float)
    q=np.asarray(second_report.get('protected_points_input_xy'),float)
    error=None
    if p.shape==(11,2) and q.shape==(11,2) and np.isfinite(p).all() and np.isfinite(q).all():
        error=float(np.percentile(np.linalg.norm(p-q,axis=1),90))
    metrics={'mask_iou':iou,'smaller_mask_containment':containment,
             'semantic_point_p90_disagreement_px':error,
             'semantic_point_p90_limit_px':PARTIAL_MASK_MAX_SEMANTIC_P90_PX,
             'minimum_partial_mask_containment':PARTIAL_MASK_MIN_CONTAINMENT}
    if iou>=.70:return dict(metrics,compatible=True,reason='overlapping_reference_masks')
    both_safe=all(report.get('mask_safety',{}).get('accepted',False) for report in (first_report,second_report))
    if containment>=PARTIAL_MASK_MIN_CONTAINMENT and error is not None and error<=PARTIAL_MASK_MAX_SEMANTIC_P90_PX and both_safe:
        return dict(metrics,compatible=True,reason='compatible_partial_reference_mask')
    return dict(metrics,compatible=False,reason='conflicting_reference_mask_geometry')


def apply_projected_mask(gray,mask):
    if gray.shape!=(SIZE,SIZE) or mask.shape!=gray.shape or gray.dtype!=np.uint8:
        raise ValueError('Expected512x512uint8 images')
    out=gray.copy();out[mask>0]=MASK_VALUE
    return out


def projected_mask_safety(polygon,ticks,pivot,footprint):
    """Mask only the visible LCD; reject any overlap with the needle's sweep.

    A correctly registered LCD may extend beyond the sensor boundary. Its
    off-screen area is not evidence of a failed registration. Intersect the
    projection with the current sensor footprint before checking visible area
    and analog protection; padding never becomes part of the applied mask.
    """
    polygon=np.asarray(polygon,float);ticks=np.asarray(ticks,float);pivot=np.asarray(pivot,float)
    if not np.isfinite(polygon).all() or not cv2.isContourConvex(polygon.astype(np.float32)):
        return None,{'accepted':False,'reason':'invalid_or_nonconvex_lcd_polygon'}
    area=abs(float(cv2.contourArea(polygon.astype(np.float32))))
    if not MIN_VISIBLE_LCD_AREA_PX2<=area<=SIZE*SIZE*.45:
        return None,{'accepted':False,'reason':'implausible_lcd_area','area_px2':area}
    bounds=np.asarray(footprint,float)
    if bounds.shape!=(4,2) or not np.isfinite(bounds).all() or not cv2.isContourConvex(bounds.astype(np.float32)):
        return None,{'accepted':False,'reason':'invalid_sensor_footprint'}
    frame_bounds=np.array([[0,0],[SIZE-1,0],[SIZE-1,SIZE-1],[0,SIZE-1]],np.float32)
    sensor_area,sensor_polygon=cv2.intersectConvexConvex(bounds.astype(np.float32),frame_bounds)
    if sensor_polygon is None or sensor_area<=0:
        return None,{'accepted':False,'reason':'invalid_sensor_footprint'}
    visible_area,visible_polygon=cv2.intersectConvexConvex(polygon.astype(np.float32),sensor_polygon)
    visibility={'area_px2':area,'visible_area_px2':float(visible_area),
                'visible_fraction':float(np.clip(visible_area/area,0,1)),
                'partial_visibility':bool(area-visible_area>1e-3),
                'visibility_policy':MASK_VISIBILITY_POLICY,
                'minimum_visible_area_px2':MIN_VISIBLE_LCD_AREA_PX2,
                'sensor_footprint_input_xy':bounds.tolist()}
    if visible_polygon is None or visible_area<MIN_VISIBLE_LCD_AREA_PX2:
        return None,dict(visibility,accepted=False,reason='insufficient_visible_lcd_area')
    visible_polygon=visible_polygon.reshape(-1,2)
    visibility['visible_lcd_polygon_input_xy']=visible_polygon.tolist()
    if ticks.shape!=(10,2) or pivot.shape!=(2,) or not np.isfinite(ticks).all() or not np.isfinite(pivot).all():
        return None,{'accepted':False,'reason':'invalid_protected_dial_geometry'}
    mask=mask_polygon(visible_polygon)
    mask=cv2.bitwise_and(mask,mask_polygon(sensor_polygon.reshape(-1,2)))
    # Letterboxing has an axis-aligned sensor footprint. Keep only pixel
    # centers inside its true bounds, even when polygon rasterization rounds.
    yy,xx=np.ogrid[:SIZE,:SIZE]
    sensor_centers=((xx>=bounds[:,0].min())&(xx<=bounds[:,0].max())
                    &(yy>=bounds[:,1].min())&(yy<=bounds[:,1].max()))
    mask[~sensor_centers]=0
    visibility['visible_mask_pixels']=int(np.count_nonzero(mask))
    if visibility['visible_mask_pixels']<MIN_VISIBLE_LCD_AREA_PX2:
        return None,dict(visibility,accepted=False,reason='insufficient_visible_lcd_area')
    protect=np.zeros_like(mask)
    # The filled fan protects all intermediate needle angles, not only the ten
    # calibration rays. It is an exclusion guard, never a speed label.
    sweep=np.concatenate([pivot[None],ticks],axis=0)
    cv2.fillPoly(protect,[np.rint(sweep).astype(np.int32)],255)
    protect=cv2.dilate(protect,np.ones((3,3),np.uint8))
    cv2.circle(protect,tuple(np.rint(pivot).astype(int)),5,255,-1)
    for point in ticks:
        cv2.circle(protect,tuple(np.rint(point).astype(int)),4,255,-1)
        cv2.line(protect,tuple(np.rint(pivot).astype(int)),tuple(np.rint(point).astype(int)),255,3)
    if np.any((mask>0)&(protect>0)):
        return None,dict(visibility,accepted=False,reason='projected_lcd_overlaps_protected_needle_or_ticks')
    return mask,dict(visibility,accepted=True,reason='safe_partial_visible_lcd' if visibility['partial_visibility'] else 'safe_projected_mask')


class MaskReference:
    def __init__(self,ident,image,polygon,calibration):
        self.id=str(ident);self.image=np.asarray(image,np.uint8)
        if self.image.shape!=(SIZE,SIZE):raise ValueError('Reference must be512x512gray')
        self.polygon=np.asarray(polygon,float)
        self.ticks=np.asarray(calibration['ticks_xy'],float)
        self.pivot=np.asarray(calibration['pivot_xy'],float)
        self.sift=cv2.SIFT_create(nfeatures=3500,contrastThreshold=.012)
        self.clahe=cv2.createCLAHE(2.,(8,8))
        region=np.zeros_like(self.image)
        cv2.polylines(region,[np.rint(self.ticks).astype(np.int32)],False,255,48)
        region[mask_polygon(self.polygon)>0]=0
        cv2.circle(region,tuple(np.rint(self.pivot).astype(int)),30,0,-1)
        self.keypoints,self.descriptors=self.sift.detectAndCompute(self.clahe.apply(self.image),region)
        if self.descriptors is None or len(self.keypoints)<12:raise ValueError('Reference has insufficient static features: '+self.id)

    def propose(self,keypoints,descriptors,footprint):
        report={'reference_id':self.id,'accepted':False,'matches':0,'inliers':0,'reason':None}
        if descriptors is None:return None,dict(report,reason='missing_current_frame_features')
        pairs=cv2.BFMatcher().knnMatch(self.descriptors,descriptors,k=2)
        good=[a for pair in pairs if len(pair)==2 for a,b in [pair] if a.distance<.72*b.distance]
        unique={}
        for match in sorted(good,key=lambda m:m.distance):unique.setdefault(match.trainIdx,match)
        good=list(unique.values());report['matches']=len(good)
        if len(good)<12:return None,dict(report,reason='insufficient_static_scale_matches')
        ref=np.float32([self.keypoints[m.queryIdx].pt for m in good])
        current=np.float32([keypoints[m.trainIdx].pt for m in good])
        cv2.setRNGSeed(812)
        h,inside=cv2.findHomography(ref,current,cv2.RANSAC,2.3,maxIters=5000,confidence=.999)
        if h is None or inside is None or not np.isfinite(h).all():return None,dict(report,reason='invalid_mask_transform')
        inside=inside.ravel().astype(bool);count=int(inside.sum())
        report.update(inliers=count,inlier_fraction=float(inside.mean()))
        if count<12:return None,dict(report,reason='insufficient_static_inliers')
        try:
            predicted=project(ref[inside],h)
            errors=np.linalg.norm(predicted-current[inside],axis=1)
            support=float(cv2.contourArea(cv2.convexHull(ref[inside])))
            nearest=np.argmin(np.linalg.norm(ref[inside,None]-self.ticks[None],axis=2),axis=1)
            bins=len(np.unique(nearest));span=int(nearest.max()-nearest.min())*20
            report.update(rmse_px=float(np.sqrt(np.mean(errors**2))),p95_error_px=float(np.percentile(errors,95)),
                          reference_support_area_px2=support,static_tick_neighborhoods=bins,speed_support_span=span)
            if report['inlier_fraction']<.4 or report['rmse_px']>1.7 or report['p95_error_px']>3.5 or support<1500 or bins<4 or span<100:
                return None,dict(report,reason='mask_registration_quality_failed')
            # Orientation check is local to the actual dial; full-frame padding
            # and unused projective extrapolation cannot force a camera pose.
            local=project([self.pivot,self.pivot+[1,0],self.pivot+[0,1]],h)
            det=float(np.linalg.det(np.stack([local[1]-local[0],local[2]-local[0]],axis=1)))
            if not .025<det<25:return None,dict(report,reason='reflected_or_degenerate_dial_projection')
            polygon=project(self.polygon,h);ticks=project(self.ticks,h);pivot=project([self.pivot],h)[0]
            mask,safety=projected_mask_safety(polygon,ticks,pivot,footprint)
            report['mask_safety']=safety
            if mask is None:return None,dict(report,reason=safety['reason'])
        except (ValueError,cv2.error,np.linalg.LinAlgError) as exc:
            return None,dict(report,reason='invalid_mask_projection',error_type=type(exc).__name__)
        report.update(accepted=True,reason='accepted',reference_to_input_h=h.tolist(),
                      lcd_polygon_input_xy=safety['visible_lcd_polygon_input_xy'],
                      projected_lcd_polygon_input_xy=polygon.tolist(),
                      protected_points_input_xy=np.concatenate([pivot[None],ticks],axis=0).tolist())
        return mask,report


class DualReferenceMasker:
    def __init__(self,references):
        self.references=list(references)
        if len(self.references)<2:raise ValueError('V5requires both straight and tilted references')
        self.sift=cv2.SIFT_create(nfeatures=5000,contrastThreshold=.012)
        self.clahe=cv2.createCLAHE(2.,(8,8))

    def process(self,gray,footprint):
        keypoints,descriptors=self.sift.detectAndCompute(self.clahe.apply(gray),None)
        proposals=[];reports=[]
        for ref in self.references:
            try:mask,report=ref.propose(keypoints,descriptors,footprint)
            except (cv2.error,ValueError,np.linalg.LinAlgError) as exc:
                mask=None;report={'reference_id':ref.id,'accepted':False,'reason':'registration_computation_failed','error':str(exc)}
            reports.append(report)
            if mask is not None and report['accepted']:proposals.append((mask,report))
        meta={'accepted':False,'reason':None,'candidates':reports,'lcd_mask_applied':False,
              'input_image_warped':False,'method':'dual_reference_SIFT_mask_projection_only',
              'mask_visibility_policy':MASK_VISIBILITY_POLICY,
              'source_labels_used':False,'previous_frame_transform_used':False}
        if not proposals:return None,None,dict(meta,reason='no_reliable_lcd_mask')
        proposals.sort(key=lambda pair:(pair[1]['inliers'],-pair[1]['rmse_px']),reverse=True)
        mask,best=proposals[0]
        if len(proposals)>1:
            agreements=[]
            for other,other_report in proposals[1:]:
                agreement=compare_reference_masks(mask,best,other,other_report)
                agreement.update(first_reference_id=best['reference_id'],second_reference_id=other_report['reference_id'])
                agreements.append(agreement)
                if not agreement['compatible']:
                    meta['candidate_mask_agreement']=agreements
                    return None,None,dict(meta,reason='ambiguous_reference_mask_projection')
                if agreement['reason']=='compatible_partial_reference_mask' and np.count_nonzero(other)>np.count_nonzero(mask):
                    mask,best=other,other_report
            meta['candidate_mask_agreement']=agreements
        partial=any(item['reason']=='compatible_partial_reference_mask' for item in meta.get('candidate_mask_agreement',[]))
        meta.update(accepted=True,reason='compatible_partial_reference_mask' if partial else 'accepted',lcd_mask_applied=True,selected_reference_id=best['reference_id'],
                    lcd_polygon_input_xy=best['lcd_polygon_input_xy'],reference_to_input_h=best['reference_to_input_h'],
                    selected_mask_safety=best.get('mask_safety'))
        return apply_projected_mask(gray,mask),mask,meta


def safe_asset(root,relative,hashes):
    if not isinstance(relative,str):raise ValueError('Missing relative asset path')
    path=(root/relative).resolve()
    expected=hashes.get(relative);expected=expected.get('sha256') if isinstance(expected,dict) else expected
    if not path.is_relative_to(root) or not path.is_file() or not expected or sha256(path)!=expected:
        raise ValueError('Missing, unsafe or corrupt protected asset: '+relative)
    return path


def load_static_assets(bundle):
    """Read static manifest/calibrations only. Never open annotations/labels."""
    root=Path(bundle).resolve();manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    hashes=manifest.get('sha256',manifest.get('files',{}))
    config=json.loads(safe_asset(root,'preprocessing_config.json',hashes).read_text(encoding='utf-8'))
    expected_config={'preprocessing_id':PREPROCESSING_ID,'color':'grayscale','input_size':[512,512],
        'lcd_mask_applied':True,'lcd_mask_value':128,
        'input_geometry':'full_frame_square_letterbox_then512_no_crop_no_canonical_warp',
        'camera_stabilization':'none','border_value':0,
        'network_normalization':'grayscale_repeated_RGB_ImageNet_mean_std'}
    if any(config.get(k)!=v for k,v in expected_config.items()):
        raise ValueError('Bundle is not the expected V5 native-view preprocessing')
    records=json.loads(safe_asset(root,'mask_references.json',hashes).read_text(encoding='utf-8'))
    if isinstance(records,dict):records=records.get('references',records.get('entries'))
    if not isinstance(records,list):raise ValueError('Invalid static mask references')
    references=[]
    for item in records:
        path=safe_asset(root,item['reference_image'],hashes)
        cal=load_calibration(safe_asset(root,item['calibration'],hashes))
        references.append(MaskReference(item['id'],cv2.imread(str(path),cv2.IMREAD_GRAYSCALE),item['lcd_polygon_xy'],cal))
    canonical=load_calibration(safe_asset(root,'calibration.json',hashes))
    return DualReferenceMasker(references),canonical,{'manifest_sha256':sha256(root/'manifest.json'),
        'preprocessing':config,'references':records,'calibration_sha256':sha256(root/'calibration.json')}


class V5Reader:
    def __init__(self,checkpoint,bundle_dir,device='cpu',*,baseline_v4=False):
        import torch
        from mixed_view_v5 import train_landmarks as trainer
        self.device=('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else str(device)
        if baseline_v4:
            from real_adaptation_v4.train_landmarks import load_model
        else:load_model=trainer.load_model
        self.model,payload=load_model(checkpoint,self.device)
        self.masker,self.calibration,assets=load_static_assets(bundle_dir)
        identity=payload.get('identity',{})
        expected_cv=identity.get('opencv_version')
        if expected_cv is None or expected_cv!=cv2.__version__:raise ValueError('OpenCV differs from checkpoint evaluation runtime')
        if payload.get('keypoint_names')!=list(LANDMARK_NAMES):raise ValueError('Unexpected landmark ordering')
        if not baseline_v4:
            if identity.get('dataset',{}).get('manifest_sha256')!=assets['manifest_sha256']:
                raise ValueError('V5checkpoint belongs to another dataset manifest')
            if trainer.PREPROCESSING_ID!=PREPROCESSING_ID:raise ValueError('Trainer and inference preprocessing IDs differ')
        for relative,expected in identity.get('implementation_hashes',{}).items():
            path=(ROOT/relative).resolve()
            if not path.is_relative_to(ROOT) or not path.is_file() or sha256(path)!=expected:
                raise ValueError('Checkpoint implementation changed: '+relative)
        self.metadata={'pipeline':'v5_fullframe_native_orientation_projected_LCD128',
            'checkpoint_sha256':sha256(checkpoint),'checkpoint_best_epoch':payload.get('best_epoch',payload.get('epoch')),
            'weight_role':'frozen_V4_weights_with_V5_preprocessing_baseline' if baseline_v4 else 'trained_V5',
            'device':self.device,'opencv':cv2.__version__,'numpy':np.__version__,'assets':assets,
            'inference_code_sha256':sha256(__file__),'speed_geometry':'frozen_V4_predicted_ticks_only',
            'geometry_code_sha256':sha256(ROOT/'real_adaptation_v4/geometry.py'),
            'input_image_warped':False,'input_roi_cropped':False,'smoothing':False,
            'mask_visibility_policy':MASK_VISIBILITY_POLICY,
            'limitation':'Current-frame indicated needle speed; startup sweeps are not road speed. No real ground truth accuracy.'}

    def predict(self,frame_bgr,include_model_input=False):
        import torch
        from PIL import Image
        from training_pilot.train_core import preprocess_image
        result={'accepted':False,'speed_kmh':None,'reason':None,'failure_stage':None,'model_evaluated':False,
            'mask_registration':None,'geometry':None,'points_input_xy':None,'points_source_xy':None,
            'availability':None,'mask_polygon_source_xy':None,'annotation_origin':'model_prediction_not_ground_truth'}
        if include_model_input:result['model_input']=None
        gray,matrix,footprint=letterbox(frame_bgr)
        masked,mask,registration=self.masker.process(gray,footprint)
        result['mask_registration']=registration
        if masked is None:
            return dict(result,reason=registration['reason'],failure_stage='lcd_mask_registration')
        if include_model_input:result['model_input']=masked.copy()
        result['mask_polygon_source_xy']=project(registration['lcd_polygon_input_xy'],np.linalg.inv(matrix)).tolist()
        tensor=preprocess_image(Image.fromarray(masked),SIZE)[None].to(self.device)
        with torch.inference_mode():out=self.model(tensor)
        points=out['points'][0].detach().float().cpu().numpy()*511
        avail=out['visibility'][0].detach().float().cpu().numpy()
        result['model_evaluated']=True
        if points.shape!=(12,2) or avail.shape!=(12,) or not np.isfinite(points).all() or not np.isfinite(avail).all():
            return dict(result,reason='invalid_model_landmarks',failure_stage='model')
        source=project(points,np.linalg.inv(matrix))
        result.update(points_input_xy=points.tolist(),points_source_xy=source.tolist(),availability=avail.tolist())
        active=avail>=.5;active[:2]=True
        height,width=frame_bgr.shape[:2]
        inside=(source[:,0]>=0)&(source[:,0]<=width-1)&(source[:,1]>=0)&(source[:,1]<=height-1)
        if not inside[active].all():return dict(result,reason='available_landmark_outside_sensor',failure_stage='geometry')
        rounded=np.clip(np.rint(points).astype(int),0,511)
        if (mask[rounded[active,1],rounded[active,0]]>0).any():
            return dict(result,reason='predicted_landmark_intersects_lcd_mask',failure_stage='mask_safety_after_prediction')
        predicted_needle=np.zeros_like(mask)
        cv2.line(predicted_needle,tuple(rounded[0]),tuple(rounded[1]),255,2)
        if np.any((predicted_needle>0)&(mask>0)):
            return dict(result,reason='predicted_needle_crosses_lcd_mask',failure_stage='mask_safety_after_prediction')
        reading=read_speed(points,visibility=avail,calibration=self.calibration)
        result.update(accepted=bool(reading['accepted']),speed_kmh=reading['speed_kmh'],reason=reading['reason'],
                      failure_stage=None if reading['accepted'] else 'geometry',geometry=reading)
        return result
