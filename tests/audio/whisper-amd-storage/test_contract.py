"""Focused refusals for the new storage boundary and report reconstruction."""
import copy
import ast
import importlib
import unittest
from unittest.mock import patch
from common import *
from storage_contract import STORAGE, validate_storage, verify_schedule, safe_member, replace_pending
from statistics_exact import exact_rows
import stage


class ContractTests(unittest.TestCase):
    def test_helpers_and_embedded_scripts_resolve_correctly(self):
        for name in ['prepare','audit','stage','collect','close','verify','finish','observe']:
            module=importlib.import_module(name)
            self.assertEqual(Path(module.__file__).parent,TOOLS)
        for name in ['stage.py','collect.py','observe.py']:
            tree=ast.parse((TOOLS/name).read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Constant) and isinstance(node.value,str) and '\n' in node.value and ('print(json.dumps' in node.value or 'terminal(' in node.value):
                    compile(PRELUDE+node.value.replace('%r','None'),name+'-remote','exec')

    def storage(self):
        return dict(base=REMOTE,filesystem='tmpfs',mount='/dev/shm',device=17,mount_device=17,root_device=1,
            free=STORAGE['preflight'],root_free=598016)

    def test_root_pressure_does_not_describe_output_device(self):
        validate_storage(self.storage(),preflight=True)
        value=self.storage();value['free']=STORAGE['remaining'];validate_storage(value)
        with self.assertRaises(AssertionError):validate_storage(value,preflight=True)

    def test_storage_refuses_wrong_device_location_and_reserve(self):
        mutations=[dict(device=1),dict(device=18),dict(filesystem='ext4'),dict(mount='/tmp'),
            dict(base='/home/vermorel/Onnx/artifacts/whisper'),dict(base=REMOTE+'/../other'),
            dict(free=STORAGE['remaining']-1),dict(root_free=-1)]
        for change in mutations:
            with self.subTest(change=change),self.assertRaises(AssertionError):validate_storage(self.storage()|change)

    def test_tar_member_refusals(self):
        self.assertEqual(safe_member('campaign/timing/00-managed/worker/result.json'),'campaign/timing/00-managed/worker/result.json')
        for name in ['/etc/passwd','../outside','a/../../b','a\\b','C:/file','a//b','a/./b','a/','']:
            with self.subTest(name=name),self.assertRaises(AssertionError):safe_member(name)

    def test_four_fresh_workers(self):
        runs=[dict(phase='timing',family='whisper',engine=e,child=dict(pid=100+i,birth=1000+i)) for i,e in enumerate(['managed','ort','ort','managed'])]
        verify_schedule(runs)
        duplicate=copy.deepcopy(runs);duplicate[3]['child']=duplicate[0]['child']
        for bad in [runs[:3],runs[::-1][1:]+runs[:1],duplicate]:
            with self.assertRaises(AssertionError):verify_schedule(bad)

    def test_current_scoreboard_section_is_preserved(self):
        original=(ROOT/'BENCHMARK.md').read_text(encoding='utf8');section='### Audio: matched AMD Whisper baseline\n\nVerified fixture.\n\n'
        changed=replace_pending(original,section)
        self.assertEqual(changed.split(section)[0],original.split("**Whisper's matched AMD comparison is incomplete:**")[0])
        self.assertEqual(changed.split('### Audio: Windows Microsoft ONNX Runtime baselines')[1],original.split('### Audio: Windows Microsoft ONNX Runtime baselines')[1])
        with self.assertRaises(AssertionError):replace_pending(changed,section)

    def test_live_predecessor_refused_before_remote_work(self):
        prior=dict(complete=False,code=None,supervisor=dict(pid=42,birth=123))
        with patch.object(stage,'read',return_value=prior),patch.object(stage,'ssh') as remote:
            with self.assertRaises(AssertionError):stage.e5_gate()
            remote.assert_not_called()
        prior['complete']=True;prior['code']=0
        with patch.object(stage,'read',return_value=prior),patch.object(stage,'absent',return_value=False):
            with self.assertRaises(AssertionError):stage.e5_gate()

    def workers(self):
        workers=[];cases=[dict(name='a',samples=16000),dict(name='b',samples=16000)]
        for engine in ['managed','ort','ort','managed']:
            records=[]
            for iteration in range(4):
                for index,case in enumerate(cases):
                    ticks=(index+1)*(10 if engine=='managed' else 5)
                    records.append(dict(name=case['name'],**{'pass':iteration},start_ticks=100,end_ticks=100+ticks,frequency=10,seconds=-999))
            workers.append(dict(engine=engine,records=records))
        return workers,cases

    def test_integer_tick_reconstruction_ignores_derived_seconds(self):
        workers,cases=self.workers();rows=exact_rows(workers,cases)
        self.assertEqual(rows[0]['managed']['seconds'],3)
        self.assertEqual(rows[0]['ort']['seconds'],1.5)
        self.assertEqual(rows[0]['ratio'],2)
        self.assertEqual(rows[0]['managed']['rtf'],1.5)
        self.assertEqual(rows[0]['ort']['rtf'],.75)
        self.assertEqual(len(rows),3)

    def test_missing_or_duplicated_measured_call_is_refused(self):
        for duplicate in [False,True]:
            workers,cases=self.workers();workers[0]['records'].pop()
            if duplicate:workers[0]['records'].append(workers[0]['records'][-1])
            with self.assertRaises(AssertionError):exact_rows(workers,cases)

    def test_real_saved_conformance_and_damaged_records(self):
        base=OLD/'collected';manifest=read(base/'manifests/whisper.json')
        value=read(base/'campaign/conformance/00-whisper-managed/worker/result.json')
        validate_records(value,manifest,'conformance')
        for mutate in [lambda v:v['records'].pop(),lambda v:v['records'][0].update(ownership=False),lambda v:v.update(held_outputs_unchanged=False)]:
            damaged=copy.deepcopy(value);mutate(damaged)
            with self.assertRaises(AssertionError):validate_records(damaged,manifest,'conformance')


if __name__=='__main__':
    unittest.main(verbosity=2)
