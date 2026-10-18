"""Download the public LCD dataset using pinned manifests and SHA-256 hashes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin
from urllib.request import urlopen
import hashlib
import json
import os
import shutil
import tempfile
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _download(url, path, expected_sha, expected_size=None):
    path = Path(path)
    if path.is_file() and sha256(path) == expected_sha:
        return path
    temporary = path.with_name(path.name + '.partial')
    with urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
        shutil.copyfileobj(response, stream, length=1024 * 1024)
    if expected_size is not None and temporary.stat().st_size != expected_size:
        raise ValueError(f'Unexpected byte count for {path.name}')
    if sha256(temporary) != expected_sha:
        raise ValueError(f'SHA-256 mismatch for {path.name}')
    os.replace(temporary, path)
    return path


def safe_extract(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            part = PurePosixPath(info.filename)
            if (part.is_absolute() or '..' in part.parts or '\\' in info.filename
                    or ':' in info.filename or not part.parts or part.parts[0] != 'dataset'):
                raise ValueError(f'Unsafe archive member: {info.filename}')
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlinks are not supported')
            if not (destination / info.filename).resolve().is_relative_to(destination):
                raise ValueError('Archive member escapes destination')
        bundle.extractall(destination)


def download_dataset(manifest_url, output_dir, expected_manifest_sha256):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _download(manifest_url, output_dir / 'download_manifest.json',
                              expected_manifest_sha256)
    manifest = json.loads(manifest_path.read_text())
    parts = manifest['parts']
    for part in parts:
        if Path(part['name']).name != part['name'] or '/' in part['name'] or '\\' in part['name']:
            raise ValueError('Part names must be plain filenames')
    def fetch(part):
        result = _download(urljoin(manifest_url, part['name']), output_dir / part['name'],
                           part['sha256'], part['bytes'])
        print(f'Verified LCD dataset part: {result.name}', flush=True)
        return result
    with ThreadPoolExecutor(max_workers=4) as executor:
        files = list(executor.map(fetch, parts))
    archive = output_dir / manifest['archive_name']
    if archive.name != manifest['archive_name']:
        raise ValueError('Archive name must be a filename')
    if not archive.is_file() or sha256(archive) != manifest['archive_sha256']:
        temporary = archive.with_suffix(archive.suffix + '.partial')
        with temporary.open('wb') as target:
            for path in files:
                with path.open('rb') as source:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
        if temporary.stat().st_size != manifest['archive_bytes'] or sha256(temporary) != manifest['archive_sha256']:
            raise ValueError('Combined archive verification failed')
        os.replace(temporary, archive)
    destination = output_dir / ('extracted_' + manifest['archive_sha256'][:16])
    marker = destination / '.verified_archive.json'
    if not marker.exists():
        if destination.exists():
            raise FileExistsError(f'Incomplete extraction requires inspection: {destination}')
        with tempfile.TemporaryDirectory(dir=output_dir, prefix='lcd_extract_') as temporary:
            temporary = Path(temporary)
            safe_extract(archive, temporary)
            if sha256(temporary / 'dataset' / 'manifest.json') != manifest['dataset_manifest_sha256']:
                raise ValueError('Dataset manifest mismatch')
            (temporary / '.verified_archive.json').write_text(json.dumps(manifest, indent=2))
            os.replace(temporary, destination)
    if sha256(destination / 'dataset' / 'manifest.json') != manifest['dataset_manifest_sha256']:
        raise ValueError('Extracted dataset manifest mismatch')
    return destination / 'dataset'
