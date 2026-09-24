"""Retire four closed VM duplicates with fully verified local originals."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('retirement', HERE / 'retire_redundant_padding_exports.py')
retirement = importlib.util.module_from_spec(spec); spec.loader.exec_module(retirement)
retirement.BASE = retirement.ROOT / 'artifacts/parakeet-inclusive-packing-precontract-retention-20260924'
retirement.TARGETS = [
    ('parakeet-inclusive-packing-build-amd-20260924', 'lokad-parakeet-inclusive-packing-build-20260924',
     '50f2a3a8a2ebbe20d41315bc0be4242e24fb7f3eb74e78533091a750791a4078', ['inventory/instructions.json']),
    ('pyannote-winograd-register-transform-build-amd-20260923', 'lokad-pyannote-winograd-register-transform-build-20260923',
     '0da29afbb3f75e92aeb66cc5ffc2667984f24f034d6aa4275b7e3bd31d21bbec', ['inventory/instructions.json']),
    ('graph-startup-diagnostic-amd-20260923', 'lokad-graph-startup-diagnostic-20260923',
     'c2f4961396ec577711ef3ac48e629e901414a21a327594921b59fb015bda9a1f',
     ['a-capture/capture.nettrace', 'b-capture/capture.nettrace']),
]

if __name__ == '__main__':
    retirement.main()
    retirement.save(retirement.BASE / 'configuration.json', dict(adapter=retirement.pin(Path(__file__)),
        implementation=retirement.pin(HERE / 'retire_redundant_padding_exports.py'), targets=retirement.TARGETS,
        closed=retirement.pin(retirement.BASE / 'closed.json'),
        scope='Four duplicate VM output files only. Complete local instructions, raw traces, archives and all immutable VM inputs retained.'))
