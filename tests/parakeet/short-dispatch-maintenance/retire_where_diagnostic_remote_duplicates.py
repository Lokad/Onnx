"""Retire eight M60 VM duplicates while retaining complete verified local evidence."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('export_retirement',HERE/'retire_redundant_padding_exports.py')
retirement=importlib.util.module_from_spec(spec);spec.loader.exec_module(retirement)
retirement.BASE=retirement.ROOT/'artifacts/parakeet-where-diagnostic-remote-retention-20260924'
retirement.TARGETS=[
    ('parakeet-dense-scalar-where-control-codegen-amd-20260924','lokad-parakeet-dense-scalar-where-control-codegen-20260924',
     '9b4ad473ff373bea1b50ecb20d9b3f7896fb8b84dbbfb0c71c70692f55290e1a',
     [f'{job}/{name}' for job in ['current-screen0-512','candidate-screen1-512','candidate-screen2-512','current-screen3-512']
      for name in ['clocks.jsonl','result.json']]),
]

if __name__=='__main__':
    retirement.main()
    retirement.save(retirement.BASE/'configuration.json',dict(adapter=retirement.pin(Path(__file__)),
        implementation=retirement.pin(HERE/'retire_redundant_padding_exports.py'),targets=retirement.TARGETS,
        closed=retirement.pin(retirement.BASE/'closed.json'),
        scope='Eight VM output duplicates only. All local clocks/results, archives, disassembly and immutable VM inputs retained.'))
