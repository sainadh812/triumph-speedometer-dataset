"""Exercise the actual multipart join/checksum/safe-extraction path offline."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from real_adaptation_v4 import notebook_helpers as nh


class MultipartTests(unittest.TestCase):
    def test_verified_parts_join_extract_and_corruption_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'fixture.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('dataset2000/annotations.json', '{"fixture":true}')
            data = archive.read_bytes()
            parts = []
            for index, content in enumerate((data[:60], data[60:])):
                path = root / f'fixture.zip.part{index:02d}'
                path.write_bytes(content)
                parts.append({'filename': path.name, 'sha256': nh.sha256(path), 'bytes': len(content)})
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'archive_name': archive.name, 'sha256': nh.sha256(archive),
                'bytes': len(data), 'parts': parts, 'root': 'dataset2000'}))
            def offline_download(url, destination, digest):
                source = root / url.rsplit('/', 1)[-1]
                if nh.sha256(source) != digest:
                    raise ValueError('Source part SHA256 mismatch')
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                return destination
            with patch.object(nh, 'download_verified', offline_download):
                output = nh.prepare_dataset(manifest, 'https://example.invalid', root / 'work')
                self.assertEqual(json.loads((output / 'annotations.json').read_text()), {'fixture': True})
                (root / parts[0]['filename']).write_bytes(b'wrong')
                with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                    nh.prepare_dataset(manifest, 'https://example.invalid', root / 'work2')

    def test_duplicate_part_names_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            part = {'filename': 'same.part00', 'sha256': hashlib.sha256(b'x').hexdigest(), 'bytes': 1}
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'archive_name': 'fixture.zip', 'sha256': '0' * 64,
                'bytes': 2, 'parts': [part, part], 'root': 'dataset2000'}))
            def fake_download(url, destination, digest):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b'x')
                return destination
            with patch.object(nh, 'download_verified', fake_download):
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    nh.prepare_dataset(manifest, 'https://example.invalid', root / 'work')


if __name__ == '__main__':
    unittest.main()
