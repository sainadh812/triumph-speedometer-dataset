"""Compile generated notebook cells and verify pins; execute no notebook code."""
import hashlib
import json
from pathlib import Path
import unittest

from build_notebook import RUNTIME_FILES, build

REPO = Path(__file__).resolve().parent.parent


class NotebookTests(unittest.TestCase):
    def test_generated_cells_compile_without_running(self):
        notebook = build(REPO, 'a' * 40)
        self.assertEqual(notebook['nbformat'], 4)
        self.assertNotIn('accelerator', notebook['metadata'])
        code = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
        self.assertEqual(len(code), 2)
        for cell in code:
            compile(''.join(cell['source']), '<generated-' + cell['id'] + '>', 'exec')
            self.assertEqual(cell['outputs'], [])
            self.assertIsNone(cell['execution_count'])

    def test_runtime_files_are_hash_pinned_in_both_cells(self):
        notebook = build(REPO, 'b' * 40)
        paths = ['dashboard_joint_v1/' + name for name in RUNTIME_FILES]
        paths.append('learned_lcd_v1/drive_api_video_v1/drive_api_store.py')
        for cell in notebook['cells']:
            if cell['cell_type'] != 'code':
                continue
            source = ''.join(cell['source'])
            self.assertIn('b' * 40, source)
            for path in paths:
                self.assertIn(hashlib.sha256((REPO / path).read_bytes()).hexdigest(), source)

    def test_public_template_uses_placeholders_and_private_config_does_not_add_secret(self):
        public = json.dumps(build(REPO, 'c' * 40))
        self.assertIn('SET_PRIVATE_DATASET_FILE_ID', public)
        self.assertIn('SET_PRIVATE_WORKSPACE_FOLDER_ID', public)
        self.assertNotIn('@gmail.com', public)
        private = build(REPO, 'c' * 40, {'dataset_reference': {'id': 'private-test-file', 'sha256': 'd' * 64, 'size': 100},
                                       'workspace_id': 'private-test-folder', 'run_id': 'test_run'})
        text = json.dumps(private)
        self.assertIn('private-test-file', text)
        self.assertIn("from_colab_secret('TRIUMPH_DRIVE_OAUTH')", text)
        self.assertNotIn('refresh_token', text)
        self.assertNotIn('client_secret', text)


if __name__ == '__main__':
    unittest.main()
