"""Reject damaged native application results and Linux resource/identity evidence."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
import audit
import remote


class EvidenceTests(unittest.TestCase):
    def fixture(self):
        identity = dict(schema=1, complete=True, started=1, ended=10,
            supervisor=dict(pid=100,start=1000,affinity='0'),
            limits=dict(rss=13*1024**3,seconds=1800,available_memory=256*1024**2),runs=[])
        samples = {}
        terminal = [dict(pid=100,start=1000)]
        for i,name in enumerate(['managed','cli-connected','cli-token-limit']):
            pid = 101+i
            member = dict(pid=pid,start=1001+i,group=pid,affinity='2',state='S',rss=100,cpu_seconds=.2)
            identity['runs'].append(dict(name=name,code=0,seconds=1,started=2+2*i,ended=3+2*i,
                pid=pid,start=1001+i,samples=2,peak_rss=100,members={str(pid):1001+i}))
            samples[name] = [dict(seconds=s,members=[copy.deepcopy(member)],available_memory=1024**3) for s in (.2,.7)]
            terminal.append(dict(pid=pid,start=1001+i))
        collection = dict(complete=True,code=0,checkout='172181fc5ab4eb2bdc2eb7f37e80d25e482a0887',terminal_processes=terminal)
        return identity,samples,collection

    def test_process_evidence(self):
        self.assertEqual(len(audit.process_audit(*self.fixture())),3)
        mutations = [
            lambda j,s,c:j.update(complete=False),
            lambda j,s,c:j.update(error='worker failed'),
            lambda j,s,c:j['supervisor'].update(affinity='2'),
            lambda j,s,c:j['limits'].update(rss=20*1024**3),
            lambda j,s,c:j['runs'].pop(),
            lambda j,s,c:j['runs'][0].update(code=1),
            lambda j,s,c:j['runs'][0].update(seconds=1800),
            lambda j,s,c:j['runs'][1].update(started=2.5),
            lambda j,s,c:j['runs'][0].update(samples=3),
            lambda j,s,c:j['runs'][0].update(peak_rss=99),
            lambda j,s,c:j['runs'][0].update(start=999),
            lambda j,s,c:s['managed'][1].update(seconds=.1),
            lambda j,s,c:s['managed'][0].update(available_memory=256*1024**2-1),
            lambda j,s,c:s['managed'][0]['members'][0].update(affinity='0-3'),
            lambda j,s,c:s['managed'][0]['members'][0].update(group=999),
            lambda j,s,c:s['managed'][0]['members'][0].update(start=999),
            lambda j,s,c:s['managed'][0]['members'][0].update(rss=13*1024**3),
            lambda j,s,c:s['managed'][1]['members'][0].update(cpu_seconds=.1),
            lambda j,s,c:s['managed'][0]['members'].append(copy.deepcopy(s['managed'][0]['members'][0])),
            lambda j,s,c:c['terminal_processes'].pop(),
            lambda j,s,c:c.update(code=2),
            lambda j,s,c:c.update(checkout='different'),
        ]
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index):
                args = self.fixture()
                mutate(*args)
                with self.assertRaises((AssertionError,KeyError)):
                    audit.process_audit(*args)

    def test_runtime_guards(self):
        _,samples,_ = self.fixture()
        sample = samples['managed'][0]
        remote.check_sample(sample,101,1001)
        for field,value in [('seconds',1800),('available_memory',0)]:
            damaged = copy.deepcopy(sample)
            damaged[field] = value
            with self.assertRaises(AssertionError):remote.check_sample(damaged,101,1001)
        for field,value in [('rss',13*1024**3),('affinity','0'),('start',1002),('group',102)]:
            damaged = copy.deepcopy(sample)
            damaged['members'][0][field] = value
            with self.assertRaises(AssertionError):remote.check_sample(damaged,101,1001)

    def test_application_evidence(self):
        base = Path(__file__).resolve().parents[3]/'artifacts/parakeet-recording-20260919'
        read = lambda name:json.loads((base/name).read_text(encoding='utf-8'))
        native,windows,inputs = read('native/result.json'),read('managed/result.json'),read('inputs/inputs.json')
        canonical = lambda values:[v.replace('-∞','-Infinity') for v in values]
        validate = lambda managed:audit.application_audit(managed,native,windows,inputs,lambda *args:None,canonical,lambda case:None,{})
        self.assertEqual(len(validate(copy.deepcopy(windows))),10)
        mutations = [
            lambda j:j['cases'].pop(),
            lambda j:j['cases'][0].update(repeat=True),
            lambda j:j['cases'][0].update(seconds=float('nan')),
            lambda j:j['cases'][0]['result'].update(text='changed'),
            lambda j:j['cases'][0]['result'].update(processed_seconds=1),
            lambda j:j['cases'][0]['result']['windows'][0]['decoding']['token_ids'].__setitem__(0,0),
            lambda j:j['cases'][0]['result']['windows'][0]['decoding']['frame_indices'].__setitem__(0,-1),
            lambda j:j['refusals'].pop(),
            lambda j:j['concurrent'].pop(),
            lambda j:j.update(ownership=False),
            lambda j:j.update(flags={'LOKAD_ONNX_TEST':'1'}),
        ]
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index):
                damaged = copy.deepcopy(windows)
                mutate(damaged)
                with self.assertRaises(AssertionError):validate(damaged)

    def test_safe_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            self.assertEqual(remote.safe_path(base,'inputs/file.npy'),(base/'inputs/file.npy').resolve())
            for path in ('../outside','inputs/../../outside','/absolute','inputs\\outside','.'):
                with self.subTest(path=path),self.assertRaises(AssertionError):remote.safe_path(base,path)


if __name__ == '__main__':
    unittest.main()
