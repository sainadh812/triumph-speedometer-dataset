"""Procedural RPM-only training examples from one reviewed, empty TRAIN LCD.

No learned model or image-generation service is used. Pixels containing RPM
numbers and the toothed scale remain source pixels. The synthetic scalar is the
chosen rendering target, NOT a measured physical engine RPM. The renderer is an
approximation of LCD segmented fill and must not be used as real test truth.
"""
from __future__ import annotations
import copy
import cv2
import numpy as np
try:
    from .preprocess import SIZES, RPM_NAMES, letterbox_matrix, project
except ImportError:
    from preprocess import SIZES, RPM_NAMES, letterbox_matrix, project

RENDERER_VERSION = 'reviewed-empty-lcd-segments-v1'
# Hand-traced in a 640x320 semantic LCD rectification of training source night_00.
LEFT_BOUNDARY = np.array([[97,10],[109,30],[126,60],[136,80],[144,100],[151,120],
 [157,140],[161,160],[165,180],[168,200],[170,220],[171,240],[171,260],
 [169,280],[167,300],[164,320]], np.float32)
RIGHT_TOOTH_PATH = np.array([[175,14],[168,24],[181,34],[181,47],[197,61],
 [191,74],[205,86],[200,99],[215,111],[208,124],[222,138],[214,151],
 [224,162],[219,176],[229,189],[221,202],[229,213],[222,226],
 [229,239],[222,252],[226,264],[219,279],[224,290],[214,303],[220,316]],np.float32)
# Anchor row is numeral CENTER after rectification, not raw-image horizontal row.
ANCHORS = np.array([[220,316],[226,264],[229,213],[224,162],
                    [215,111],[197,61],[175,14]],np.float32)
FIXED_TOP_PRINT = np.array([[96,-4],[142,-4],[157,15],[117,40]],np.float32)
VALUES = np.arange(0,12001,2000,dtype=np.float32)
CROP_MATRIX = letterbox_matrix([64,-16,360,336], SIZES['rpm'])
GEOMETRY = {'coordinate_space':'semantic_LCD_640x320',
 'left_boundary':LEFT_BOUNDARY.tolist(),'right_tooth_path':RIGHT_TOOTH_PATH.tolist(),
 'anchors_0_2_4_6_8_10_12':ANCHORS.tolist(),'fill_inset_pixels':3,
 'protected_top_print':FIXED_TOP_PRINT.tolist(),
 'rpm_band':[64,-16,360,336], 'segment_pitch_canonical_pixels':8,
 'segment_note':'Observed real-image pixel pitch; not claimed factory RPM quantization.',
 'provenance':'Manual raw-image trace of training night_00, reviewed against real training night_11 fill. No validation/test image used.'}

def _canonical(source_bgr, annotation, source_split):
    if source_split != 'train' or annotation.get('id') != 'night_00':
        raise ValueError('Renderer is reviewed only for empty TRAIN source night_00')
    if annotation.get('state') != 'off' or annotation['rpm_points']['bar_end']['visible']:
        raise ValueError('Base must be reviewed empty/off LCD with no visible filled endpoint')
    q=np.asarray(annotation.get('lcd_rectification_corners',annotation['lcd_corners']),np.float32)
    if q.shape!=(4,2) or not np.isfinite(q).all():
        raise ValueError('Reviewed semantic quadrilateral required')
    image=cv2.cvtColor(source_bgr,cv2.COLOR_BGR2GRAY) if source_bgr.ndim==3 else source_bgr
    h=cv2.getPerspectiveTransform(q,np.array([[0,0],[640,0],[640,320],[0,320]],np.float32))
    canonical=cv2.warpPerspective(image,h,(640,320),flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,borderValue=128)
    # Keep actual surrounding source pixels above/below LCD in final crop.
    model_base=cv2.warpPerspective(image,CROP_MATRIX@h,SIZES['rpm'],flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,borderValue=128)
    return canonical,model_base

def _point_for_rpm(rpm):
    # Smooth common scale-edge calibration; exact scalar is this piecewise path.
    return np.array([np.interp(rpm,VALUES,ANCHORS[:,0]),
                     np.interp(rpm,VALUES,ANCHORS[:,1])],np.float32)

def _render_canonical(base, rpm, seed):
    if not np.isfinite(rpm) or not 0 <= rpm <= 12000:
        raise ValueError('Synthetic RPM must be finite within 0..12000')
    rng=np.random.default_rng(seed)
    scale=3
    alpha=np.zeros((320*scale,640*scale),np.uint8)
    # Each segment is sampled along traced boundaries. Insets protect original
    # thin outlines/tooth ink. Gap reproduces separate LCD fill segments.
    endpoint_y=float(_point_for_rpm(rpm)[1])
    for y_bottom in np.arange(316,endpoint_y,-8):
        y_top=max(endpoint_y,y_bottom-8)
        gap=min(1.4,(y_bottom-y_top)*.22)
        ys=np.linspace(y_top,y_bottom-gap,max(3,int((y_bottom-y_top)*2)+2))
        left_y=np.minimum(316,ys+4)
        left=np.interp(left_y,LEFT_BOUNDARY[:,1],LEFT_BOUNDARY[:,0])+3
        right=np.interp(ys,RIGHT_TOOTH_PATH[:,1],RIGHT_TOOTH_PATH[:,0])-3
        poly=np.vstack([np.c_[left,left_y],np.c_[right[::-1],ys[::-1]]])
        cv2.fillPoly(alpha,[np.round(poly*scale).astype(np.int32)],255)
    alpha=cv2.resize(alpha,(640,320),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    alpha=cv2.GaussianBlur(alpha,(3,3),.55)
    protected=np.zeros((320,640),np.uint8)
    cv2.fillPoly(protected,[np.round(FIXED_TOP_PRINT).astype(np.int32)],255)
    protected=cv2.dilate(protected,np.ones((3,3),np.uint8))
    alpha[protected>0]=0
    alpha[317:]=0
    # Real training N11 measured median gray contrast about63; sampled variation
    # covers the softened darker stripe center and source reflection changes.
    contrast=float(rng.uniform(61,82))
    shade=np.clip(base.astype(np.float32)-contrast,55,205)
    rendered=base*(1-alpha)+shade*alpha
    return np.round(rendered).astype(np.uint8),alpha

def _compose_input(base,model_base,rendered):
    delta=rendered.astype(np.float32)-base.astype(np.float32)
    delta=cv2.warpPerspective(delta,CROP_MATRIX,SIZES['rpm'],flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    return np.clip(np.round(model_base.astype(np.float32)+delta),0,255).astype(np.uint8)

def render_example(source_bgr, annotation, rpm, *, source_split='train', seed=0):
    """One raw-source-derived 256x384 grayscale image and eight model landmarks."""
    base,model_base=_canonical(source_bgr,annotation,source_split)
    rendered,mask=_render_canonical(base,float(rpm),seed)
    image=_compose_input(base,model_base,rendered)
    points=project(np.vstack([ANCHORS,_point_for_rpm(float(rpm))]),CROP_MATRIX)
    visible=[True]*8
    if float(rpm)==0:
        points[7]=0
        visible[7]=False  # Identical real off/empty source has no observed endpoint.
    return {'image':image,'points':points.tolist(),'visible':visible,
            'synthetic_rpm':float(rpm),'source_id':annotation['id'],'split':'train',
            'synthetic':True,'renderer_version':RENDERER_VERSION,
            'synthetic_target_note':'Chosen display-rendering scalar; not physical RPM ground truth.',
            'geometry':copy.deepcopy(GEOMETRY),
            'canonical_fill_mask':np.round(mask*255).astype(np.uint8)}

def generate_examples(source_bgr, source_annotation, *, source_split='train',count=240,seed=240):
    """Yield balanced 0..12000 RPM variants; caller applies geometry/photometry.

    Every descendant MUST keep source_id night_00 and split train. No default
    image output or filesystem mutations. Count240 provides61 display levels.
    """
    if count<1:raise ValueError('count must be positive')
    base,model_base=_canonical(source_bgr,source_annotation,source_split)
    levels=np.resize(np.arange(0,12001,200),int(count))
    rng=np.random.default_rng(seed);rng.shuffle(levels)
    for index,rpm in enumerate(levels):
        rendered,mask=_render_canonical(base,float(rpm),seed+index)
        image=_compose_input(base,model_base,rendered)
        points=project(np.vstack([ANCHORS,_point_for_rpm(float(rpm))]),CROP_MATRIX)
        visible=[True]*8
        if float(rpm)==0:
            points[7]=0
            visible[7]=False
        yield {'image':image,'points':points.tolist(),'visible':visible,
          'synthetic_rpm':float(rpm),'source_id':source_annotation['id'],'split':'train',
          'synthetic':True,'renderer_version':RENDERER_VERSION,
          'synthetic_target_note':'Chosen display-rendering scalar; not physical RPM ground truth.',
          'geometry':copy.deepcopy(GEOMETRY)}
