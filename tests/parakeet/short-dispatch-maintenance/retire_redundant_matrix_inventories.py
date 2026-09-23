"""Retire remote inventory copies from five terminal, locally retained matrix builds."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('export_retirement', HERE / 'retire_redundant_padding_exports.py')
retirement = importlib.util.module_from_spec(spec); spec.loader.exec_module(retirement)
retirement.BASE = retirement.ROOT / 'artifacts/parakeet-redundant-matrix-inventory-retention-20260923'
retirement.TARGETS = [
    ('parakeet-first-use-kernels-build-amd-20260923', 'lokad-parakeet-first-use-kernels-build-20260923',
     '2fb4e3e587ab463a965d7cd4290ffe3f37674182bb47b7ee529d040825c3f243', ['inventory/instructions.json']),
    ('parakeet-short-dispatch-build-amd-v2-20260923', 'lokad-parakeet-short-dispatch-build-v2-20260923',
     '6832658ff8963554674356817689f72e04bc113cad09b2522bc218cd7d427ed5', ['inventory/instructions.json']),
    ('parakeet-short-dispatch-build-amd-v3-20260923', 'lokad-parakeet-short-dispatch-build-v3-20260923',
     '0a2d96454588d8e90d246ccbf91c1069d0e1efdd88710b16985dda89f42b3d39', ['inventory/instructions.json']),
    ('parakeet-short-wide-pack-build-amd-20260923', 'lokad-parakeet-short-wide-pack-build-20260923',
     '23746895d6c89a887f55cc7e15a2a19907d9753e3b348a9959e515aa6c060578', ['inventory/instructions.json']),
    ('parakeet-wide-per-call-build-amd-20260923', 'lokad-parakeet-wide-per-call-build-20260923',
     '3359287fcbde0cd4742744270437ce2b76ea53ea9107e878d72d41d869cc290d', ['inventory/instructions.json']),
]

if __name__ == '__main__':
    retirement.main()
    retirement.save(retirement.BASE / 'configuration.json', dict(
        adapter=retirement.pin(Path(__file__)), implementation=retirement.pin(HERE / 'retire_redundant_padding_exports.py'),
        targets=retirement.TARGETS, closed=retirement.pin(retirement.BASE / 'closed.json')))
