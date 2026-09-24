"""Check the two qualified protocols against every complete retained case."""
import ast
import json
from pathlib import Path
import sys
import unittest
from statistics import summarize
from statistics_base import summarize as baseline
from statistics_e5 import summarize as warmed_e5
from protocol import CASES,ORDER

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
# Exercise the real pure command builder without importing the Linux monitor.
tree=ast.parse((TOOLS/'remote.py').read_text(encoding='utf8'))
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='command_for')
namespace=dict(BASE=ROOT/'unused-command-test',DOTNET='/home/vermorel/.dotnet/dotnet',sys=sys)
exec(compile(ast.Module(body=[function],type_ignores=[]),'remote.py:command_for','exec'),namespace)
command_for=namespace['command_for']


class Dispatch(unittest.TestCase):
    def test_all_retained_case_scores_are_exact(self):
        for key in CASES:
            namespace='e5-warmed-qualification-amd-20260924' if key=='e5-30tok' else 'parakeet-validated-composition-graphs-amd-20260924'
            folder=ROOT/'artifacts'/namespace/'collected'
            reports={role:json.loads((folder/('timing-'+key+'-'+role)/'output/result.json').read_text()) for role in ORDER}
            expected=(warmed_e5 if key=='e5-30tok' else baseline)(reports)
            self.assertEqual(expected,summarize(reports),key)
            for role in ['current','candidate','ort']:
                command,build=command_for('timing-'+key+'-'+role+'-a',{})
                self.assertFalse(build)
                if role=='ort':
                    self.assertEqual(Path(command[2]).name,'native-e5.py' if key=='e5-30tok' else 'native.py')
                else:
                    self.assertEqual(Path(command[1]).parent.parent.name,'runtimes-e5' if key=='e5-30tok' else 'runtimes')

    def test_mixed_or_unknown_case_identity_is_rejected(self):
        from test_statistics import rows
        reports=rows()
        for row in reports.values():row['key']='resnet50'
        reports['ort-a']['key']='e5-30tok'
        with self.assertRaises(AssertionError):summarize(reports)
        for row in reports.values():row['key']='unknown'
        with self.assertRaises(AssertionError):summarize(reports)


if __name__=='__main__':unittest.main()
