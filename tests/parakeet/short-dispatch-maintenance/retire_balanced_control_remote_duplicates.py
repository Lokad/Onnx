"""Retire eight closed M61 control VM outputs, with complete verified local copies."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('export_retirement',HERE/'retire_redundant_padding_exports.py')
retirement=importlib.util.module_from_spec(spec);spec.loader.exec_module(retirement)
retirement.BASE=retirement.ROOT/'artifacts/parakeet-balanced-control-remote-retention-20260924'
retirement.TARGETS=[
    ('parakeet-dense-scalar-where-balanced-control-amd-20260924','lokad-parakeet-dense-scalar-where-balanced-control-20260924',
     'e9c296b8cc4c3be86a8e8447225e5254fae2f264741da8dcc128ade8d6d6db9c',
     [f'{job}/{name}' for job in ['current-screen0-512','candidate-screen1-512','candidate-screen2-512','current-screen3-512']
      for name in ['clocks.jsonl','result.json']]),
]

if __name__=='__main__':
    retirement.main()
    retirement.save(retirement.BASE/'configuration.json',dict(adapter=retirement.pin(Path(__file__)),
        implementation=retirement.pin(HERE/'retire_redundant_padding_exports.py'),targets=retirement.TARGETS,
        closed=retirement.pin(retirement.BASE/'closed.json'),
        scope='Eight VM output duplicates only. Full local clocks, results, collection archive and all immutable VM inputs retained.'))
