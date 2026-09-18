"""Package transfer must preserve code and reject changes underneath a run."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('recreate3_package_state', ROOT / 'python_helper/package_state.py')
package_state = importlib.util.module_from_spec(spec)
spec.loader.exec_module(package_state)
sys.path.insert(0, str(ROOT / 'runtime'))
from gsm8k_experiment import archive


class PackageStateTests(unittest.TestCase):
    def make_package(self, root):
        root.mkdir()
        (root / 'runtime').mkdir()
        (root / 'provenance').mkdir()
        (root / 'runtime/run.py').write_text('print("fixture")\n')
        (root / 'provenance/package_manifest.json').write_text(json.dumps({'files': {
            'runtime/run.py': hashlib.sha256((root / 'runtime/run.py').read_bytes()).hexdigest()}}))

    def test_stage_preserves_existing_cli_configs_and_rejects_code_change(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            package, runtime = root / 'package', root / 'local/runtime'
            self.make_package(package)
            package_state.stage(package, runtime)
            configs = runtime / '.experiment_cli/configs'
            configs.mkdir(parents=True)
            (configs / 'config.json').write_text('{}')
            package_state.stage(package, runtime)
            self.assertEqual((configs / 'config.json').read_text(), '{}')
            (runtime / 'run.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Existing local runtime changed'):
                package_state.stage(package, runtime)
            self.assertEqual((runtime / 'run.py').read_text(), 'changed')

    def test_extra_runtime_code_and_corrupt_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'package'
            self.make_package(root)
            (root / 'runtime/extra.py').write_text('unexpected')
            with self.assertRaisesRegex(ValueError, 'unexpected or missing'):
                package_state.verify(root)
            (root / 'runtime/extra.py').unlink()
            (root / 'runtime/run.py').write_text('corrupt')
            with self.assertRaisesRegex(ValueError, 'differs'):
                package_state.verify(root)

    def test_manifest_cannot_escape_package(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / 'package'
            self.make_package(root)
            for name in ('../outside.py', str(Path(td) / 'outside.py'), '.'):
                with self.assertRaises(ValueError):
                    package_state.safe_path(root, name)

    def make_snapshot(self, root, name):
        output = root / 'source' / name
        output.mkdir(parents=True)
        identity = json.dumps({'seed_directory': name, 'fingerprint': 'unchanged'}).encode()
        (output / 'manifest.json').write_bytes(identity)
        checkpoint = output / 'arms/proxy/checkpoint.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'checkpoint for ' + name.encode())
        checkpoint.with_suffix('.sha256.json').write_text(json.dumps({
            'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}))
        archive_root = root / 'archive'
        with mock.patch.dict(os.environ, {'RECREATE3_ARCHIVE_ROOT': str(archive_root)}):
            archive.archive_output(output)
        return output, archive_root

    def test_restore_all_seed_and_parallel_worker_names_without_collisions(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            names = ('seed_42', 'seed_42_prepare', 'seed_42_proxy', 'seed_42_judge',
                     'seed_42_knn_static', 'seed_0', 'seed_007')
            for name in names:
                self.make_snapshot(root, name)
            destination = root / 'local/runs'
            package_state.restore(ROOT / 'runtime', root / 'archive', destination)
            self.assertEqual({p.name for p in destination.iterdir()}, set(names))
            for name in names:
                with self.subTest(name=name):
                    source = root / 'source' / name
                    restored = destination / name
                    source_files = {p.relative_to(source): p.read_bytes()
                                    for p in source.rglob('*') if p.is_file()}
                    restored_files = {p.relative_to(restored): p.read_bytes()
                                      for p in restored.rglob('*') if p.is_file()}
                    self.assertEqual(restored_files, source_files)

    def test_restore_ignores_lookalikes_and_unpublished_folders(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            names = ('seed_42_', 'seed_42_prox', 'seed_42_judge30b', 'seed_42_knn_static_30b',
                     'seed_42_prepare_more', 'seed_42_proxy_more', 'seed_42_arms',
                     'seed_-1', 'seed_x', 'seed_42.0', 'seed_42_prepare.csv', 'anotherseed_42')
            for name in names:
                self.make_snapshot(root, name)
            (root / 'archive/seed_43_prepare').mkdir()
            (root / 'archive/seed_44_proxy').write_text('not a directory')
            destination = root / 'local/runs'
            package_state.restore(ROOT / 'runtime', root / 'archive', destination)
            self.assertEqual(list(destination.iterdir()), [])

    def test_restore_retains_existing_local_work_for_seed_and_worker(self):
        for name in ('seed_42', 'seed_42_proxy'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                source, archive_root = self.make_snapshot(root, name)
                destination = root / 'local/runs'
                existing = destination / name
                existing.mkdir(parents=True)
                (existing / 'local-only.txt').write_bytes(b'keep this work')
                package_state.restore(ROOT / 'runtime', archive_root, destination)
                self.assertEqual((existing / 'manifest.json').read_bytes(),
                                 (source / 'manifest.json').read_bytes())
                retained = list((destination.parent / 'retained_local').iterdir())
                self.assertEqual(len(retained), 1)
                self.assertTrue(retained[0].name.startswith(name + '-'))
                self.assertEqual((retained[0] / 'local-only.txt').read_bytes(), b'keep this work')
                self.assertFalse((existing / 'local-only.txt').exists())

    def test_corrupt_snapshot_keeps_existing_local_work_untouched(self):
        for name in ('seed_42', 'seed_42_knn_static'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                _, archive_root = self.make_snapshot(root, name)
                seed_archive = archive_root / name
                latest = json.loads((seed_archive / 'latest.json').read_text())
                snapshot = seed_archive / 'snapshots' / latest['snapshot']
                (snapshot / 'output/manifest.json').write_bytes(b'corrupted')
                destination = root / 'local/runs'
                existing = destination / name
                existing.mkdir(parents=True)
                (existing / 'local-only.txt').write_bytes(b'keep this work')
                with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                    package_state.restore(ROOT / 'runtime', archive_root, destination)
                self.assertEqual((existing / 'local-only.txt').read_bytes(), b'keep this work')
                self.assertEqual(list(destination.iterdir()), [existing])
                self.assertFalse((destination.parent / 'retained_local').exists())

    def test_missing_archive_preserves_local_work(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            destination = root / 'local/runs'
            existing = destination / 'seed_42_prepare'
            existing.mkdir(parents=True)
            (existing / 'manifest.json').write_bytes(b'local preparation')
            package_state.restore(ROOT / 'runtime', root / 'missing_archive', destination)
            self.assertEqual((existing / 'manifest.json').read_bytes(), b'local preparation')


if __name__ == '__main__':
    unittest.main()
