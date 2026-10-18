"""Package only reviewed dataset assets; original videos and weights stay private."""
from pathlib import Path
import json
import zipfile
from learned_lcd_v1.download_dataset import sha256


def package(root):
    root = Path(root).resolve()
    dataset = root / 'dataset'
    required = ['manifest.json', 'sources.json', 'samples.jsonl']
    if not all((dataset / p).is_file() for p in required) or (dataset / 'BUILD_INCOMPLETE.json').exists():
        raise FileNotFoundError('Complete, reviewed LCD dataset required before packaging')
    manifest = json.loads((dataset / 'manifest.json').read_text())
    publication = root / 'publication'
    publication.mkdir(exist_ok=True)
    archive = root / 'learned_lcd_v1_dataset.zip'
    allowed = {'.png', '.jpg', '.jpeg', '.json', '.jsonl', '.md', '.txt', '.csv'}
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as bundle:
        for path in sorted(dataset.rglob('*')):
            if path.is_file():
                if path.suffix.lower() not in allowed:
                    raise ValueError(f'Unexpected dataset asset: {path}')
                bundle.write(path, path.relative_to(root).as_posix())
    parts = []
    with archive.open('rb') as stream:
        index = 0
        while block := stream.read(45 * 1024 * 1024):
            path = publication / f'{archive.name}.part{index:02d}'
            path.write_bytes(block)
            parts.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
            index += 1
    rows = [json.loads(line) for line in (dataset / 'samples.jsonl').read_text().splitlines() if line.strip()]
    counts = {split: sum(r['split'] == split for r in rows) for split in sorted({r['split'] for r in rows})}
    result = {'schema_version': 1, 'archive_name': archive.name, 'archive_sha256': sha256(archive),
              'archive_bytes': archive.stat().st_size, 'dataset_manifest_sha256': sha256(dataset / 'manifest.json'),
              'counts': counts, 'parts': parts}
    (publication / 'download_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    package(Path(__file__).parent)
