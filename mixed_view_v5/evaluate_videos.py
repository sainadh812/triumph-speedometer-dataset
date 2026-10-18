"""Colab-only current-frame V5 video evaluation; no training or local inference.

30FPS is the sampling/playback timeline, not a real-time throughput claim.
Source video is shown in full. Numeric speed is omitted when mask registration,
model input, framing, or predicted-landmark geometry is unreliable.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from mixed_view_v5.inference import V5Reader,sha256
from real_adaptation_v4.infer_video import probe_video,build_timeline

FPS=30
BOARD=(1280,960)


def require_colab():
    try:available=importlib.util.find_spec('google.colab') is not None
    except ModuleNotFoundError:available=False
    if not available:raise RuntimeError('Video/model evaluation is authorized in Google Colab only. Upload the private videos there and run this script in the notebook.')


def json_safe(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items() if k!='model_input'}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    return value


def text(image,label,x,y,scale=.6,color=(225,232,240),thickness=1):
    cv2.putText(image,str(label),(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,color,thickness,cv2.LINE_AA)


def wrapped(image,label,x,y,width=43):
    line=''
    for word in str(label).replace('_',' ').split():
        if len(line)+len(word)+1>width:
            text(image,line,x,y,.48);y+=22;line=word
        else:line=(line+' '+word).strip()
    if line:text(image,line,x,y,.48)
    return y


def display_label(video,index,weight_role):
    """File/presentation label only; never passed to the prediction API."""
    match=re.search(r'(\d{1,2})\.(\d{2})\.(\d{2})\s*(AM|PM)',Path(video).stem,re.IGNORECASE)
    source=f'clip{index}'
    if match:
        hour,minute,second=int(match[1]),int(match[2]),int(match[3])
        if 1<=hour<=12 and minute<60 and second<60:
            hour=hour%12+(12 if match[4].upper()=='PM' else 0)
            source=f'{hour:02d}{minute:02d}{second:02d}'
    role='before' if weight_role=='frozen_V4_weights_with_V5_preprocessing_baseline' else 'after'
    return f'{role}_{source}'


def render(frame,result,index,total,timestamp,weight_role,checkpoint_epoch=None,clip_label=''):
    board=np.full((BOARD[1],BOARD[0],3),(29,24,20),np.uint8)
    source=frame.copy();points=result.get('points_source_xy')
    color=(110,240,155) if result['accepted'] else (65,180,250)
    if points is not None:
        points=np.asarray(points,float)
        if points.shape==(12,2) and np.isfinite(points).all() and np.max(np.abs(points))<10000:
            integer=np.rint(points).astype(int)
            cv2.line(source,tuple(integer[0]),tuple(integer[1]),color,2,cv2.LINE_AA)
            availability=np.asarray(result['availability'])
            for j,point in enumerate(integer):
                c=color if availability[j]>=.5 else (150,150,150)
                cv2.circle(source,tuple(point),3,c,1,cv2.LINE_AA)
                if j>=2:text(source,str((j-2)*20),point[0]+4,point[1]-4,.34,c)
    polygon=result.get('mask_polygon_source_xy')
    if polygon is not None:
        cv2.polylines(source,[np.rint(polygon).astype(np.int32)],True,(245,210,45),2,cv2.LINE_AA)
    h,w=source.shape[:2];scale=min(760/w,840/h)
    nw,nh=round(w*scale),round(h*scale);x=20+(760-nw)//2;y=95+(840-nh)//2
    board[y:y+nh,x:x+nw]=cv2.resize(source,(nw,nh),interpolation=cv2.INTER_AREA)
    text(board,'TRIUMPH | Mixed-view speed model',20,35,.93,thickness=2)
    text(board,'Full source frame | current-frame needle + semantic ticks',20,67,.58)
    role_label=(f'BASELINE: frozen V4 epoch {checkpoint_epoch} weights, V5 input pipeline'
                if weight_role=='frozen_V4_weights_with_V5_preprocessing_baseline'
                else f'TRAINED: V5 best checkpoint epoch {checkpoint_epoch}')
    text(board,role_label,20,87,.43)
    text(board,clip_label,815,45,.58)
    text(board,'INDICATED SPEED',815,116,.67)
    value=f"{result['speed_kmh']:.1f} km/h" if result['accepted'] else 'Unavailable'
    text(board,value,815,170,1.12,color,2)
    reason='Geometric consistency passed; not proof of accuracy' if result['accepted'] else result['reason']
    wrapped(board,reason,815,210)
    stage=result.get('failure_stage') or 'none'
    wrapped(board,'Rejected stage: '+stage,815,280)
    registration=result.get('mask_registration') or {}
    wrapped(board,'LCD mask reference: '+str(registration.get('selected_reference_id','none')),815,330)
    model_input=result.get('model_input')
    if model_input is not None:
        board[390:790,830:1230]=cv2.cvtColor(cv2.resize(model_input,(400,400),interpolation=cv2.INTER_AREA),cv2.COLOR_GRAY2BGR)
    else:
        text(board,'No model input: mask rejected',825,555,.57,color)
    text(board,'Actual 512x512 input; LCD128; no pose warp',815,820,.46)
    text(board,f'Frame {index+1}/{total} | {timestamp:.2f}s | 30FPS playback',815,854,.49)
    text(board,'Cyan: LCD mask | Green/amber: predicted points',815,884,.43)
    text(board,'No smoothing, stale reading, or assumed zero.',815,914,.45)
    text(board,'Indicated needle speed; startup sweeps are not road speed.',20,950,.43)
    return board


def run_video(video,destination,reader,label):
    require_colab()
    video,destination=Path(video).resolve(),Path(destination).resolve()
    if destination.exists():raise FileExistsError('Preserving existing evaluation: '+str(destination))
    destination.mkdir(parents=True)
    info=probe_video(video);indices,times=build_timeline(info['timestamps'],info['duration'],FPS)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():raise RuntimeError('Cannot decode uploaded video')
    intermediate=destination/'annotated_intermediate.avi'
    output=destination/(label+'_V5_30fps.mp4')
    writer=cv2.VideoWriter(str(intermediate),cv2.VideoWriter_fourcc(*'MJPG'),FPS,BOARD)
    if not writer.isOpened():raise RuntimeError('Cannot create inspection video')
    selected=set(np.linspace(0,len(indices)-1,6,dtype=int).tolist())
    rows=[];previews=[];cursor=-1;last=None;start=time.perf_counter()
    try:
        with (destination/'predictions.jsonl').open('x',encoding='utf-8') as sink:
            for index,wanted in enumerate(indices):
                fresh=False
                while cursor<wanted:
                    okay,frame=cap.read()
                    if not okay:raise RuntimeError('Source decode ended early')
                    cursor+=1;fresh=True
                if fresh:last=reader.predict(frame,include_model_input=True)
                if last is None:raise RuntimeError('Missing current source frame')
                row={'output_frame':index,'source_frame':int(wanted),'time_s':float(times[index]),
                     'source_time_s':float(info['timestamps'][wanted]),'same_source_frame_repeated':not fresh,
                     'prediction':json_safe(last)}
                sink.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
                board=render(frame,last,index,len(indices),float(times[index]),reader.metadata['weight_role'],
                             reader.metadata['checkpoint_best_epoch'],label)
                writer.write(board)
                if index in selected:
                    cv2.imwrite(str(destination/f'preview_{index:04d}.jpg'),board)
                    if last.get('model_input') is not None:cv2.imwrite(str(destination/f'input_{index:04d}.png'),last['model_input'])
                    previews.append(cv2.resize(board,(640,480),interpolation=cv2.INTER_AREA))
                if (index+1)%60==0 or index+1==len(indices):
                    sink.flush();print(json.dumps({'clip':label,'frames':index+1,'total':len(indices),'elapsed_s':round(time.perf_counter()-start,1)}),flush=True)
            while cursor<len(info['timestamps'])-1:
                if not cap.read()[0]:raise RuntimeError('Unexpected missing source frame')
                cursor+=1
            if cap.read()[0]:raise RuntimeError('Source has more frames than its probed timeline')
    finally:cap.release();writer.release()
    elapsed=time.perf_counter()-start
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-n','-i',str(intermediate),'-i',str(video),
        '-map','0:v:0','-map','1:a?','-c:v','libx264','-preset','medium','-crf','18','-pix_fmt','yuv420p',
        '-c:a','aac','-b:a','128k','-af','asetpts=PTS-STARTPTS','-t',f'{len(rows)/FPS:.9f}',
        '-movflags','+faststart',str(output)],check=True)
    check=cv2.VideoCapture(str(output));decoded=0
    while True:
        okay,picture=check.read()
        if not okay:break
        if decoded in selected:cv2.imwrite(str(destination/f'encoded_preview_{decoded:04d}.jpg'),picture)
        decoded+=1
    fps=check.get(cv2.CAP_PROP_FPS);check.release()
    if decoded!=len(rows) or abs(fps-FPS)>1e-6:raise RuntimeError('Encoded frame count/FPS mismatch')
    if abs(decoded/fps-info['duration'])>.5/FPS+1e-6:raise RuntimeError('Encoded duration mismatch')
    if len(previews)==6:cv2.imwrite(str(destination/'contact_sheet.jpg'),np.vstack([np.hstack(previews[:3]),np.hstack(previews[3:])]))
    with (destination/'predictions.csv').open('x',newline='',encoding='utf-8') as sink:
        fields=['output_frame','source_frame','time_s','speed_kmh','accepted','failure_stage','reason','mask_reference']
        writer_csv=csv.DictWriter(sink,fieldnames=fields);writer_csv.writeheader()
        for row in rows:
            p=row['prediction'];writer_csv.writerow({**{k:row[k] for k in fields[:3]},
                **{k:p[k] for k in fields[3:7]},'mask_reference':(p['mask_registration'] or {}).get('selected_reference_id')})
    summary={'status':'complete_and_fully_decoded','input_filename':video.name,'input_sha256':sha256(video),
        'output':str(output),'output_sha256':sha256(output),'source_frames':len(info['timestamps']),
        'output_frames':decoded,'fps':fps,'duration_s':decoded/fps,'source_duration_s':info['duration'],
        'distinct_source_frames_attempted':len(set(indices.tolist())),
        'distinct_source_frames_model_evaluated':sum(not r['same_source_frame_repeated'] and r['prediction']['model_evaluated'] for r in rows),
        'output_frames_with_model_evaluation':sum(r['prediction']['model_evaluated'] for r in rows),
        'processing_seconds':elapsed,
        'offline_processing_fps':len(rows)/elapsed,'accepted':sum(r['prediction']['accepted'] for r in rows),
        'failure_stages':dict(Counter(r['prediction']['failure_stage'] or 'accepted' for r in rows)),
        'reasons':dict(Counter(r['prediction']['reason'] for r in rows)),
        'reader_metadata':json_safe(reader.metadata),'source_audio_preserved_if_present':'AAC transcode, aligned to start and video duration',
        'ground_truth_accuracy':None,'acceptance_is_accuracy':False,'training_performed':False,
        'evaluation_location':'Google Colab','no_stale_reading_fallback':True,'smoothing':False,
        'sampling':'30FPS nearest source timestamp; identical-source repeats share the same prediction',
        'script_sha256':sha256(__file__)}
    (destination/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    shutil.copy2(__file__,destination/'evaluate_videos.py')
    print(json.dumps({k:summary[k] for k in ('output','output_frames','accepted','failure_stages')},indent=2),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video',action='append',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda','auto'],default='auto')
    parser.add_argument('--baseline-v4',action='store_true',help='Explicit frozen V4weights evaluated using identical V5preprocessing')
    args=parser.parse_args();require_colab()
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1);cv2.setNumThreads(2)
    reader=V5Reader(args.checkpoint,args.dataset,args.device,baseline_v4=args.baseline_v4)
    summaries=[]
    for i,video in enumerate(args.video,1):
        label=display_label(video,i,reader.metadata['weight_role'])
        summaries.append(run_video(video,args.output_root/f'clip{i}',reader,label))
    (args.output_root/'evaluation_summary.json').write_text(json.dumps(summaries,indent=2,allow_nan=False)+'\n',encoding='utf-8')


if __name__=='__main__':main()
