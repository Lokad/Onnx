"""Route retained complete observations through the exact established scorers."""
import ast
from pathlib import Path
import sys
import unittest
from protocol import CASES,ORDER,read
from statistics import summarize
from statistics_base import summarize as normal
from statistics_e5 import summarize as long
from statistics_short import summarize as short
from prepare import ROOT,GRAPH,SHORT

TOOLS=Path(__file__).resolve().parent
tree=ast.parse((TOOLS/'remote.py').read_text(encoding='utf8'))
function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='command_for')
namespace=dict(BASE=ROOT/'unused-command-test',DOTNET='/home/vermorel/.dotnet/dotnet',sys=sys)
exec(compile(ast.Module(body=[function],type_ignores=[]),'command-for','exec'),namespace)
command_for=namespace['command_for']


class Dispatch(unittest.TestCase):
    def reports(self,key,folder):
        return {role:read(folder/'collected'/('timing-'+key+'-'+role)/'output/result.json') for role in ORDER}

    def test_all_retained_scores_and_every_worker_route(self):
        for key in CASES:
            folder,scorer=(SHORT,short) if key=='e5-8tok' else (GRAPH,long if key=='e5-30tok' else normal)
            reports=self.reports(key,folder)
            self.assertEqual(summarize(reports),scorer(reports))
            runtime,native=('runtimes-short','native-short.py') if key=='e5-8tok' else ('runtimes-e5','native-e5.py') if key=='e5-30tok' else ('runtimes','native.py')
            for mode in ['verify','timing']:
                for role in ['current','candidate','ort']:
                    command,build=command_for(f'{mode}-{key}-{role}'+('' if mode=='verify' else '-a'),{})
                    self.assertFalse(build)
                    if role=='ort':self.assertEqual(Path(command[2]).name,native)
                    else:self.assertEqual(Path(command[1]).parent.parent.name,runtime)

    def test_old_short_prefix_is_rejected(self):
        with self.assertRaises(AssertionError):summarize(self.reports('e5-8tok',GRAPH))

    def test_mixed_case_identity_is_rejected(self):
        reports=self.reports('e5-8tok',SHORT);reports['ort-a']['key']='resnet50'
        with self.assertRaises(AssertionError):summarize(reports)

    def test_removed_final_clock_is_rejected(self):
        reports=self.reports('e5-8tok',SHORT);reports['ort-a']['clocks'].pop()
        with self.assertRaises(AssertionError):summarize(reports)


if __name__=='__main__':unittest.main()
