"""Retire the terminal M50 build cache after verifying the protected offline feed."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cache_retirement', HERE / 'retire_padding_package_caches.py')
retirement = importlib.util.module_from_spec(spec); spec.loader.exec_module(retirement)
retirement.BASE = retirement.ROOT / 'artifacts/parakeet-isolated-short-kernels-cache-retention-20260923'
retirement.NAMES = ['lokad-parakeet-isolated-short-kernels-build-v2-20260923']
retirement.PROOFS = [('parakeet-isolated-short-kernels-build-amd-v2-20260923',
                      '0adb72e2eae376df7d4f3dd6eb7f4a97b15dd57c5d4c3fa962f8a4cd3a9c45c6')]

if __name__ == '__main__':
    retirement.main()
    retirement.save(retirement.BASE / 'configuration.json', dict(
        adapter=retirement.pin(Path(__file__)), implementation=retirement.pin(HERE / 'retire_padding_package_caches.py'),
        roots=retirement.NAMES, proofs=retirement.PROOFS, closed=retirement.pin(retirement.BASE / 'closed.json')))
