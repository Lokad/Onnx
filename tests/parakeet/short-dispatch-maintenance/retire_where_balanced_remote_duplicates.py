"""Retire four closed M61 diagnostic VM outputs after full local verification."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('export_retirement',HERE/'retire_redundant_padding_exports.py')
retirement=importlib.util.module_from_spec(spec);spec.loader.exec_module(retirement)
retirement.BASE=retirement.ROOT/'artifacts/parakeet-where-balanced-remote-retention-20260924'
retirement.TARGETS=[
    ('parakeet-dense-scalar-where-balanced-codegen-amd-20260924','lokad-parakeet-dense-scalar-where-balanced-codegen-20260924',
     'ef63ec76aaf45d9df155de015b660b606e6e106446b5e6916c2cd3c048942457',
     [f'{job}/{name}' for job in ['current-screen0-512','candidate-screen1-512'] for name in ['clocks.jsonl','result.json']]),
]

if __name__=='__main__':
    retirement.main()
    retirement.save(retirement.BASE/'configuration.json',dict(adapter=retirement.pin(Path(__file__)),
        implementation=retirement.pin(HERE/'retire_redundant_padding_exports.py'),targets=retirement.TARGETS,
        closed=retirement.pin(retirement.BASE/'closed.json'),
        scope='Four VM output duplicates only. Full local clocks, results, collection archive, native bodies and all immutable VM inputs retained.'))
