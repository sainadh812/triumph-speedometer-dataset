"""Verified multipart dataset download for the mixed-view V5 Colab notebook."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import uuid
from concurrent.futures import ThreadPoolExecutor

from real_adaptation_v3.notebook_helpers import download_verified, require_digest, sha256, safe_extract


def prepare_dataset(manifest_path, base_url, work_directory):
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    filename = manifest['archive_name']
    if not isinstance(filename, str) or re.fullmatch(r'[A-Za-z0-9_.-]+\.zip', filename) is None:
        raise ValueError('Expected plain dataset ZIP filename')
    expected = require_digest(manifest['sha256'])
    parts = manifest['parts']
    if not isinstance(parts, list) or not 1 <= len(parts) <= 64:
        raise ValueError('Invalid dataset part list')
    if manifest.get('root') != 'dataset':
        raise ValueError('Expected dataset archive root')
    root = Path(work_directory)
    root.mkdir(parents=True, exist_ok=True)
    names = set()
    paths, total_bytes, downloads = [], 0, []
    for part in parts:
        name = part['filename']
        if (not isinstance(name, str) or re.fullmatch(r'[A-Za-z0-9_.-]+', name) is None
                or name in {'.', '..'} or name.casefold() in names):
            raise ValueError('Invalid/duplicate dataset part filename')
        names.add(name.casefold())
        size = part['bytes']
        if type(size) is not int or not 0 < size <= 50 * 1024 ** 2:
            raise ValueError('Dataset part exceeds expected size')
        total_bytes += size
        path = root / name
        downloads.append((name, path, part['sha256'], size))
        paths.append(path)
    if total_bytes > 2 * 1024 ** 3:
        raise ValueError('Dataset archive exceeds expected size')
    if manifest.get('bytes') != total_bytes:
        raise ValueError('Manifest total archive size differs from summed parts')
    def fetch(record):
        name,path,digest,size=record
        result=download_verified(base_url.rstrip('/')+'/'+name,path,digest)
        if result.stat().st_size!=size:
            raise ValueError('Dataset part size differs')
        print('Verified dataset part:',name,flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(fetch,downloads))
    target = root / filename
    if target.exists():
        if target.stat().st_size != total_bytes or sha256(target) != expected:
            raise ValueError('Existing joined dataset archive differs')
    else:
        temporary = root / ('.join_' + uuid.uuid4().hex + '.partial')
        try:
            with temporary.open('xb') as destination:
                for path in paths:
                    with path.open('rb') as source:
                        while chunk := source.read(1024 * 1024):
                            destination.write(chunk)
                destination.flush()
                os.fsync(destination.fileno())
            if temporary.stat().st_size != total_bytes or sha256(temporary) != expected:
                raise ValueError('Joined dataset archive SHA256 mismatch')
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return safe_extract(target, root / 'extracted', expected, expected_root='dataset')
