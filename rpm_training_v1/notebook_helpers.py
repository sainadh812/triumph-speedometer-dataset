"""Verified download and extraction helpers for the RPM Colab pilot."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import re
import shutil
import stat
import urllib.request
import zipfile


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def download_verified(url, destination, expected_sha256):
    destination=Path(destination)
    if destination.is_file() and sha256(destination)==expected_sha256:
        return destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    pending=destination.with_name(destination.name+'.partial')
    request=urllib.request.Request(url,headers={'User-Agent':'Triumph-RPM-training-v1'})
    with urllib.request.urlopen(request,timeout=120) as response,pending.open('wb') as target:
        shutil.copyfileobj(response,target)
    if sha256(pending)!=expected_sha256:
        raise ValueError('Downloaded checksum mismatch: '+destination.name)
    os.replace(pending,destination)
    return destination


def prepare_dataset(manifest_path,base_url,work):
    manifest=json.loads(Path(manifest_path).read_text())
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    def safe_name(name):
        return isinstance(name,str) and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',name) and name not in ('.','..')
    for key in ('archive_name','root'):
        if not safe_name(manifest[key]): raise ValueError('Unsafe manifest '+key)
    if not manifest['parts']: raise ValueError('No archive parts')
    if len({e['filename'] for e in manifest['parts']})!=len(manifest['parts']):
        raise ValueError('Duplicate archive parts')
    for entry in manifest['parts']:
        name=entry['filename']
        if not safe_name(name):
            raise ValueError('Unsafe part filename')
    archive=work/manifest['archive_name']
    if not archive.is_file() or sha256(archive)!=manifest['sha256']:
        pending=archive.with_name(archive.name+'.partial')
        with pending.open('wb') as output:
            for entry in manifest['parts']:
                part=download_verified(base_url+'/'+entry['filename'],work/entry['filename'],entry['sha256'])
                if part.stat().st_size!=entry['bytes']:
                    raise ValueError('Part size mismatch')
                with part.open('rb') as source: shutil.copyfileobj(source,output)
        if pending.stat().st_size!=manifest['bytes'] or sha256(pending)!=manifest['sha256']:
            raise ValueError('Reassembled archive checksum mismatch')
        os.replace(pending,archive)
    extract=work/('extracted_'+manifest['sha256'][:16])
    marker=extract/'.verified_archive.json'
    if marker.is_file():
        cached=json.loads(marker.read_text())
        if cached.get('archive_sha256')!=manifest['sha256']:
            raise ValueError('Cached extraction marker mismatch')
        if not cached.get('files_sha256'): raise ValueError('Cached extraction has no file checksums')
        for name,digest in cached['files_sha256'].items():
            path=(extract/name).resolve()
            if not path.is_relative_to(extract.resolve()) or not path.is_file() or sha256(path)!=digest:
                raise ValueError('Cached extracted data checksum mismatch')
    if not marker.is_file():
        extract.mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            names=set()
            for entry in z.infolist():
                name=PurePosixPath(entry.filename)
                if name.is_absolute() or '..' in name.parts or '\\' in entry.filename or ':' in entry.filename:
                    raise ValueError('Unsafe archive entry')
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError('Archive symlink is unsupported')
                if not name.parts or name.parts[0]!=manifest['root']:
                    raise ValueError('Unexpected archive root')
                canonical=name.as_posix().rstrip('/').casefold()
                if canonical in names: raise ValueError('Duplicate or case-colliding archive entry')
                names.add(canonical)
            z.extractall(extract)
            verified_files={entry.filename:sha256(extract/entry.filename) for entry in z.infolist() if not entry.is_dir()}
        marker.write_text(json.dumps({'archive_sha256':manifest['sha256'],'files_sha256':verified_files}))
    data=extract/manifest['root']
    if not (data/'labels.jsonl').is_file(): raise ValueError('Dataset labels missing after extraction')
    return data
