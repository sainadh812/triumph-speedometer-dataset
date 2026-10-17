"""Package completed RPM training data as checksum-protected GitHub-sized parts."""
from pathlib import Path
import json
import zipfile
from notebook_helpers import sha256

ROOT=Path(__file__).resolve().parent


def package(data):
    data=Path(data)
    if not (data/'labels.jsonl').is_file(): raise FileNotFoundError(data/'labels.jsonl')
    release=ROOT/'release';release.mkdir(exist_ok=True)
    archive=release/'rpm_synthetic_2000_v1.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(data.rglob('*')):
            if path.is_file(): z.write(path,data.name+'/'+path.relative_to(data).as_posix())
    parts=[]
    with archive.open('rb') as source:
        i=0
        while block:=source.read(45*1024**2):
            part=release/(archive.name+f'.part{i:02d}');part.write_bytes(block)
            parts.append({'filename':part.name,'sha256':sha256(part),'bytes':part.stat().st_size});i+=1
    manifest={'archive_name':archive.name,'sha256':sha256(archive),'bytes':archive.stat().st_size,'root':data.name,'parts':parts}
    if len(parts)==1:
        manifest['parts'][0]['filename']=archive.name
    (release/'data_download_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True)
    package(p.parse_args().dataset)
