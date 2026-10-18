"""Package the checked V5 data, without weights or original private videos."""
from pathlib import Path
import hashlib
import json
import zipfile

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def package(root):
    root = Path(root).resolve()
    dataset = root/'dataset'
    if (not (dataset/'annotations.json').is_file() or not (dataset/'manifest.json').is_file()
            or (dataset/'BUILD_INCOMPLETE.json').exists()):
        raise FileNotFoundError('Dataset and manifest must be complete before packaging')
    output = root/'publication'
    output.mkdir(exist_ok=True)
    archive = root/'mixed_views_landmarks_7500_v5.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for p in sorted(dataset.rglob('*')):
            if p.is_file():
                if p.suffix.lower() in {'.mp4','.pt','.pth','.avi','.mov'}:
                    raise ValueError('Private media/weights must not enter public dataset')
                z.write(p, p.relative_to(root).as_posix())
    parts=[]
    with archive.open('rb') as f:
        i=0
        while b := f.read(45*1024*1024):
            p=output/(archive.name+f'.part{i:02d}')
            p.write_bytes(b)
            parts.append(dict(filename=p.name,bytes=len(b),sha256=sha(p)))
            i+=1
    result=dict(archive_name=archive.name,root='dataset',bytes=archive.stat().st_size,
                sha256=sha(archive),parts=parts)
    (output/'data_download_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__': package(Path(__file__).parent)
