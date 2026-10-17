"""Offline dataset-download integrity checks; no network or credentials used."""
import hashlib
import json
from pathlib import Path
import shutil
import stat
import tempfile
import unittest
from unittest.mock import patch
import warnings
import zipfile

from rpm_training_v1.notebook_helpers import prepare_dataset


class DatasetHelperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='rpm_helper_audit_')
        self.base = Path(self.temporary.name).resolve()
        self.work = self.base / 'work'

    def tearDown(self):
        self.temporary.cleanup()

    def fixture(self, extra=None):
        archive = self.base / 'source.zip'
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('dataset/labels.jsonl', '{"id":"sample"}\n')
                bundle.writestr('dataset/images/sample.txt', 'sample bytes')
                if extra:
                    for name, content in extra:
                        bundle.writestr(name, content)
        blob = archive.read_bytes()
        digest = hashlib.sha256(blob).hexdigest()
        part = self.base / 'source.part00'
        part.write_bytes(blob)
        manifest = {'archive_name': 'rpm.zip', 'root': 'dataset',
                    'sha256': digest, 'bytes': len(blob),
                    'parts': [{'filename': 'rpm.zip.part00', 'bytes': len(blob), 'sha256': digest}]}
        return manifest, part

    def prepare(self, manifest, part):
        path = self.base / 'manifest.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')

        def local_download(url, destination, expected_sha256):
            self.assertEqual(expected_sha256, hashlib.sha256(part.read_bytes()).hexdigest())
            destination = Path(destination)
            # A bad manifest must be rejected before requesting any unsafe path.
            self.assertTrue(destination.resolve().is_relative_to(self.work.resolve()),
                            'Part destination escaped the work directory')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(part, destination)
            return destination

        with patch('rpm_training_v1.notebook_helpers.download_verified', side_effect=local_download):
            return prepare_dataset(path, 'https://example.invalid/pinned', self.work)

    def test_valid_archive_and_repeat_are_verified(self):
        manifest, part = self.fixture()
        data = self.prepare(manifest, part)
        self.assertTrue((data / 'labels.jsonl').is_file())
        self.assertEqual(self.prepare(manifest, part), data)

    def test_zip_parent_traversal_rejected(self):
        manifest, part = self.fixture([('dataset/../../escape.txt', 'bad')])
        with self.assertRaises(ValueError):
            self.prepare(manifest, part)
        self.assertFalse((self.work / 'escape.txt').exists())

    def test_zip_symlink_rejected(self):
        entry = zipfile.ZipInfo('dataset/link')
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        manifest, part = self.fixture([(entry, '../../escape')])
        with self.assertRaises(ValueError):
            self.prepare(manifest, part)

    def test_case_colliding_zip_members_rejected(self):
        manifest, part = self.fixture([('dataset/IMAGES/SAMPLE.txt', 'different bytes')])
        with self.assertRaises(ValueError):
            self.prepare(manifest, part)

    def test_windows_archive_path_rejected_before_write(self):
        manifest, part = self.fixture()
        manifest['archive_name'] = '..\\escape.zip'
        with self.assertRaises(ValueError):
            self.prepare(manifest, part)
        self.assertFalse((self.base / 'escape.zip').exists())

    def test_drive_relative_archive_name_rejected(self):
        manifest, part = self.fixture()
        manifest['archive_name'] = 'C:escape.zip'
        with self.assertRaises(ValueError):
            self.prepare(manifest, part)

    def test_part_colon_rejected_before_download(self):
        manifest, part = self.fixture()
        manifest['parts'][0]['filename'] = 'part:alternate_stream'
        path = self.base / 'manifest.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        with patch('rpm_training_v1.notebook_helpers.download_verified') as download:
            with self.assertRaises(ValueError):
                prepare_dataset(path, 'https://example.invalid/pinned', self.work)
            download.assert_not_called()

    def test_cached_extracted_data_tampering_rejected(self):
        manifest, part = self.fixture()
        data = self.prepare(manifest, part)
        (data / 'labels.jsonl').write_text('changed labels', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.prepare(manifest, part)

    def test_cached_marker_identity_mismatch_rejected(self):
        manifest, part = self.fixture()
        data = self.prepare(manifest, part)
        (data.parent / '.verified_archive.json').write_text(
            json.dumps({'archive_sha256': '0' * 64}), encoding='utf-8')
        with self.assertRaises(ValueError):
            self.prepare(manifest, part)

    def test_duplicate_part_names_rejected(self):
        manifest, part = self.fixture()
        manifest['parts'].append(dict(manifest['parts'][0]))
        # A valid archive must not bypass validation of the pinned manifest.
        self.work.mkdir()
        (self.work / manifest['archive_name']).write_bytes(part.read_bytes())
        with self.assertRaises(ValueError):
            self.prepare(manifest, part)


if __name__ == '__main__':
    unittest.main()
