"""Exercise the actual recording validator and reject damaged process evidence."""
from pathlib import Path
import copy
import importlib.util
import unittest
import audit
from remote import h


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        root = Path(__file__).resolve().parents[3]
        old = root/'artifacts/whisper-recording-v2-20260919'
        cls.native = h.read(old/'native-corrected/manifest.json')
        cls.windows = h.read(old/'managed/result.json')
        cls.inputs = h.read(old/'inputs/inputs.json')
        cls.short = h.read(root/'artifacts/asr-labeled-20260919/native-whisper/manifest.json')
        spec = importlib.util.spec_from_file_location('existing_recording_validator',root/'tests/whisper/recording/audit.py')
        cls.validator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.validator)
        cls.tokenizer = Tokenizer.from_file(str(root/'models/whisper-large-v3-turbo/tokenizer.json'))

    def app(self,managed):
        return audit.application_audit(managed,self.native,self.windows,self.inputs,self.short,self.validator,self.tokenizer)

    def test_application_refusals(self):
        self.assertEqual(len(self.app(copy.deepcopy(self.windows))),5)
        mutations = [
            lambda j:j['cases'].pop(),
            lambda j:j['cases'][0].update(repeat=True),
            lambda j:j['cases'][0].update(seconds=float('nan')),
            lambda j:j['cases'][0].update(pcm_sha256='changed'),
            lambda j:j['cases'][0].update(ownership=False),
            lambda j:j['cases'][0]['result'].update(text='changed'),
            lambda j:j['cases'][0]['result'].update(processed_seconds=1),
            lambda j:j['cases'][0]['result']['windows'][0].update(advanced_seconds=0),
            lambda j:j['cases'][0]['result']['windows'][0].update(start_seconds=.02),
            lambda j:j['cases'][0]['result']['segments'][0].update(end_seconds=600),
            lambda j:j['cases'][0]['result']['windows'][0]['decoding']['token_ids'].__setitem__(0,True),
            lambda j:j['cases'][0]['result']['windows'][0]['decoding'].update(average_log_probability=float('nan')),
            lambda j:j['cases'][0]['result']['windows'][0]['decoding'].update(no_speech_probability=.99,average_log_probability=-2),
            lambda j:j.update(refusals=9),
            lambda j:j.update(flags={'DOTNET_TieredCompilation':'0'}),
            lambda j:j.update(ownership=False),
            lambda j:j['empty'].update(duration_seconds=1),
            lambda j:j['silent']['windows'].pop(),
            lambda j:j['concurrent'].pop(),
            lambda j:j['short_regression']['token_ids'].append(7),
            lambda j:j['short_regression'].update(skipped_as_no_speech=1),
            lambda j:j.update(short_manifest_sha256='changed'),
        ]
        for index,change in enumerate(mutations):
            with self.subTest(index=index):
                damaged = copy.deepcopy(self.windows)
                change(damaged)
                with self.assertRaises((AssertionError,ValueError,KeyError)):self.app(damaged)

    def test_confidence_diagnostics_are_separate(self):
        changed = copy.deepcopy(self.windows)
        changed['cases'][1]['result']['windows'][0]['decoding']['no_speech_probability'] = .05
        rows = self.app(changed)
        self.assertGreater(rows[1]['maximum_windows_confidence_difference'],.01)

    def process_fixture(self):
        identity = dict(schema=1,complete=True,started=1,ended=10,supervisor=dict(pid=100,start=1000,affinity='0'),
            limits=dict(rss=27*1024**3//2,seconds=1800,available_memory=256*1024**2),runs=[])
        samples,terminal = {},[dict(pid=100,start=1000)]
        for i,name in enumerate(['managed','cli-connected']):
            pid = 101+i
            member = dict(pid=pid,start=1001+i,group=pid,affinity='2',state='S',rss=100,cpu_seconds=.2)
            identity['runs'].append(dict(name=name,code=0,seconds=1,started=2+2*i,ended=3+2*i,pid=pid,start=1001+i,
                samples=2,peak_rss=100,members={str(pid):1001+i}))
            samples[name] = [dict(seconds=s,members=[copy.deepcopy(member)],available_memory=1024**3) for s in (.2,.7)]
            terminal.append(dict(pid=pid,start=1001+i))
        return identity,samples,dict(complete=True,code=0,checkout='172181fc5ab4eb2bdc2eb7f37e80d25e482a0887',terminal_processes=terminal)

    def test_process_refusals(self):
        self.assertEqual(len(audit.process_audit(*self.process_fixture())),2)
        mutations = [
            lambda j,s,c:j.update(complete=False),
            lambda j,s,c:j.update(error='failed'),
            lambda j,s,c:j['runs'].pop(),
            lambda j,s,c:j['runs'][0].update(code=1),
            lambda j,s,c:j['runs'][0].update(seconds=1800),
            lambda j,s,c:j['runs'][1].update(started=2.5),
            lambda j,s,c:j['runs'][0].update(samples=3),
            lambda j,s,c:j['runs'][0].update(peak_rss=99),
            lambda j,s,c:j['runs'][0].update(start=999),
            lambda j,s,c:j['limits'].update(rss=20*1024**3),
            lambda j,s,c:s['managed'][1].update(seconds=.1),
            lambda j,s,c:s['managed'][0].update(available_memory=256*1024**2-1),
            lambda j,s,c:s['managed'][0]['members'][0].update(rss=27*1024**3//2),
            lambda j,s,c:s['managed'][0]['members'][0].update(affinity='0-3'),
            lambda j,s,c:s['managed'][0]['members'][0].update(group=999),
            lambda j,s,c:s['managed'][0]['members'][0].update(start=999),
            lambda j,s,c:s['managed'][1]['members'][0].update(cpu_seconds=.1),
            lambda j,s,c:s['managed'][0]['members'].append(copy.deepcopy(s['managed'][0]['members'][0])),
            lambda j,s,c:c['terminal_processes'].pop(),
            lambda j,s,c:c.update(checkout='changed'),
        ]
        for index,change in enumerate(mutations):
            with self.subTest(index=index):
                args = self.process_fixture()
                change(*args)
                with self.assertRaises((AssertionError,KeyError)):audit.process_audit(*args)

    def test_runtime_guards(self):
        _,samples,_ = self.process_fixture()
        sample = samples['managed'][0]
        h.check_sample(sample,101,1001)
        for field,value in [('seconds',1800),('available_memory',0)]:
            damaged = copy.deepcopy(sample)
            damaged[field] = value
            with self.assertRaises(AssertionError):h.check_sample(damaged,101,1001)
        for field,value in [('rss',27*1024**3//2),('affinity','0'),('group',102),('start',1002)]:
            damaged = copy.deepcopy(sample)
            damaged['members'][0][field] = value
            with self.assertRaises(AssertionError):h.check_sample(damaged,101,1001)


if __name__ == '__main__':
    unittest.main()
