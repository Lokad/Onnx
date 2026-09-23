"""Reuse the bounded, feed-verified cache retirement for terminal M49 only."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('padding_cache_retirement', HERE / 'retire_padding_package_caches.py')
retirement = importlib.util.module_from_spec(spec); spec.loader.exec_module(retirement)
retirement.BASE = retirement.ROOT / 'artifacts/parakeet-pad-first-use-cache-retention-20260923'
retirement.NAMES = ['lokad-parakeet-pad-first-use-build-20260923']
retirement.PROOFS = [('parakeet-pad-first-use-build-amd-20260923',
                      'df197b592b163d6f9333dccf6af0f50b2f3f1a6a11e5b20ef208dc7bb8447837')]

if __name__ == '__main__':
    retirement.main()
    retirement.save(retirement.BASE / 'configuration.json', dict(
        adapter=retirement.pin(Path(__file__)), implementation=retirement.pin(HERE / 'retire_padding_package_caches.py'),
        roots=retirement.NAMES, proofs=retirement.PROOFS, closed=retirement.pin(retirement.BASE / 'closed.json')))
