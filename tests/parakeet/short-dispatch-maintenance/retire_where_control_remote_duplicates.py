"""Retire closed VM duplicates after proving complete local retention."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('export_retirement', HERE / 'retire_redundant_padding_exports.py')
retirement = importlib.util.module_from_spec(spec); spec.loader.exec_module(retirement)
retirement.BASE = retirement.ROOT / 'artifacts/parakeet-where-control-remote-retention-20260924'
retirement.TARGETS = [
    ('parakeet-isolated-short-kernels-build-amd-v2-20260923', 'lokad-parakeet-isolated-short-kernels-build-v2-20260923',
     '0adb72e2eae376df7d4f3dd6eb7f4a97b15dd57c5d4c3fa962f8a4cd3a9c45c6', ['inventory/instructions.json']),
    ('parakeet-first-use-kernels-build-amd-20260923', 'lokad-parakeet-first-use-kernels-build-20260923',
     '2fb4e3e587ab463a965d7cd4290ffe3f37674182bb47b7ee529d040825c3f243', ['previous-inventory/instructions.json']),
    ('parakeet-dense-scalar-where-stability-amd-20260924', 'lokad-parakeet-dense-scalar-where-stability-20260924',
     '882ac1267e5ef80fc7b8bf2b01f3252f5b72cba8fb2fc1da2c783fa9e7e27986',
     [f'{job}/{name}' for job in ['current-screen0-512', 'candidate-screen1-512', 'candidate-screen2-512', 'current-screen3-512']
      for name in ['clocks.jsonl', 'result.json']]),
]

if __name__ == '__main__':
    retirement.main()
    retirement.save(retirement.BASE / 'configuration.json', dict(
        adapter=retirement.pin(Path(__file__)), implementation=retirement.pin(HERE / 'retire_redundant_padding_exports.py'),
        targets=retirement.TARGETS, closed=retirement.pin(retirement.BASE / 'closed.json'),
        scope='Ten VM duplicates only: two compiler exports, four journals and four results. All complete local copies, raw clocks, collection archives and immutable inputs retained.'))
