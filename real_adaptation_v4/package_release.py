"""Create a checksum-protected ZIP and GitHub-sized parts of the reviewed dataset."""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        while block := source.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def main():
    data = ROOT/'dataset2000'
    assert (data/'manifest.json').is_file(), 'Wait for completed generation'
    release = ROOT/'release'
    release.mkdir(exist_ok=True)
    name = 'daytime_joint_landmarks_2000_v4.zip'
    archive_path = release/name
    with zipfile.ZipFile(archive_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(data.rglob('*')):
            if path.is_file():
                archive.write(path, 'dataset2000/'+path.relative_to(data).as_posix())
    parts = []
    with archive_path.open('rb') as source:
        index = 0
        while block := source.read(45 * 1024**2):
            target = release/(name+f'.part{index:02d}')
            target.write_bytes(block)
            parts.append({'filename': target.name, 'sha256': digest(target), 'bytes': len(block)})
            index += 1
    manifest = {'archive_name': name, 'sha256': digest(archive_path),
        'bytes': archive_path.stat().st_size, 'root': 'dataset2000', 'parts': parts}
    (release/'data_download_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
