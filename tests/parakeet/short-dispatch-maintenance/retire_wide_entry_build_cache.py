"""Retire terminal M54's generated cache; retain its verified offline feed."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('cache_retirement', HERE / 'retire_padding_package_caches.py')
retirement = importlib.util.module_from_spec(spec); spec.loader.exec_module(retirement)
retirement.BASE = retirement.ROOT / 'artifacts/parakeet-wide-entry-cache-retention-20260923'
retirement.NAMES = ['lokad-parakeet-wide-entry-first-use-build-20260923']
retirement.PROOFS = [('parakeet-wide-entry-first-use-build-amd-20260923',
                      'da923692f2c97cbff2774006f4a3dc911aaca39e0af5b35a89639443a75ead58')]

if __name__ == '__main__':
    retirement.main()
    retirement.save(retirement.BASE / 'configuration.json', dict(
        adapter=retirement.pin(Path(__file__)), implementation=retirement.pin(HERE / 'retire_padding_package_caches.py'),
        roots=retirement.NAMES, proofs=retirement.PROOFS, closed=retirement.pin(retirement.BASE / 'closed.json')))
