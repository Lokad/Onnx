"""Retire terminal M52's generated cache; retain its verified offline feed."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cache_retirement', HERE / 'retire_padding_package_caches.py')
retirement = importlib.util.module_from_spec(spec); spec.loader.exec_module(retirement)
retirement.BASE = retirement.ROOT / 'artifacts/parakeet-wide-projection-cache-retention-20260923'
retirement.NAMES = ['lokad-parakeet-wide-projection-isolation-build-20260923']
retirement.PROOFS = [('parakeet-wide-projection-isolation-build-amd-20260923',
                      'c67b21b2e3f9d3d09f822ab075e649e232c6c42d7e7503470711d21a1bf5604d')]

if __name__ == '__main__':
    retirement.main()
    retirement.save(retirement.BASE / 'configuration.json', dict(
        adapter=retirement.pin(Path(__file__)), implementation=retirement.pin(HERE / 'retire_padding_package_caches.py'),
        roots=retirement.NAMES, proofs=retirement.PROOFS, closed=retirement.pin(retirement.BASE / 'closed.json')))
