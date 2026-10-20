import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('restore_experiments', Path(__file__).with_name('restore-experiments.py'))
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)


class ArchiveTests(unittest.TestCase):
    def test_exact_bytes_and_no_overwrite_or_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo = root / 'repo'
            subprocess.run(['git', 'init', '-q', str(repo)], check=True)
            payload = b'original\r\n\x00bytes\n'
            (repo / 'source.py').write_bytes(payload)
            subprocess.run(['git', '-C', str(repo), 'add', 'source.py'], check=True)
            subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Archive fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
            commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).decode().strip()
            manifest = dict(commit=commit, paths=['source.py'])
            destination = root / 'restored'
            self.assertEqual(1, archive.restore(repo, manifest, destination, ''))
            self.assertEqual(payload, (destination / 'source.py').read_bytes())
            with self.assertRaises(ValueError):
                archive.restore(repo, manifest, destination, '')
            for name in ('../escape', '/absolute', 'C:/drive', 'a\\b'):
                with self.assertRaises(ValueError):
                    archive.restore(repo, dict(commit=commit, paths=[name]), root / 'unsafe', '')
            self.assertFalse((root / 'unsafe').exists())


if __name__ == '__main__':
    unittest.main()
