"""Exercise retained long/short results and refuse damaged evidence."""
from pathlib import Path
import copy
import unittest
import audit
from prepare import read
from remote import h


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[3]
        base=root/'artifacts/pyannote-limit-20260919'
        cls.managed=read(base/'default/result.json')
        cls.long=read(base/'native-reference/manifest.json')
        cls.short=read(Path(read(base/'build.json')['reference'])/'manifest.json')
        cls.input_sha=read(base/'long-application-receipt.json')['input_sha256']

    def app(self,value):
        return audit.application(value,self.long,self.short,self.managed,self.input_sha)

    def test_retained_application_and_refusals(self):
        self.assertEqual(self.app(copy.deepcopy(self.managed))['windows'],591)
        mutations=[
            lambda j:j.update(passed=False),lambda j:j.update(held_unchanged=False),lambda j:j.update(input_unchanged=False),
            lambda j:j.update(input_sha256='wrong'),lambda j:j.update(samples=9600001),lambda j:j.update(copies=19),
            lambda j:j['refusals'].pop(),lambda j:j.update(full_request_seconds=float('nan')),
            lambda j:j['result'].update(Status=1),lambda j:j['result'].update(Windows=590),
            lambda j:j['result']['Intervals'].pop(),lambda j:j['result']['ExclusiveIntervals'][0].update(Start=600),
            lambda j:j['result']['Speakers'][0]['Centroid'].__setitem__(0,float('nan')),
            lambda j:j['result']['Speakers'][0]['Centroid'].__setitem__(0,100.),
            lambda j:j['result']['Speakers'][0].update(Speaker=1),
            lambda j:j['result']['Speakers'][0].update(HasEmbedding=False),
            lambda j:j['recovery']['Intervals'][0].update(End=30),
            lambda j:j['empty'].update(Windows=1),lambda j:j.update(recovery_centroid_error=1.),
        ]
        for index,change in enumerate(mutations):
            with self.subTest(index=index):
                value=copy.deepcopy(self.managed);change(value)
                with self.assertRaises((AssertionError,ValueError,TypeError)):self.app(value)

    def fixture(self):
        member=dict(pid=102,start=200,group=102,affinity='2',state='R',rss=4096,cpu_seconds=.1)
        samples=[dict(seconds=t,members=[copy.deepcopy(member)],available_memory=1024**3) for t in (.1,.6)]
        identity=dict(schema=1,complete=True,supervisor=dict(pid=101,start=100,affinity='0'),
            limits=dict(rss=8*1024**3,seconds=1800,available_memory=256*1024**2),started=1,ended=4,
            runs=[dict(name='managed',code=0,seconds=1,started=2,ended=3,samples=2,peak_rss=4096,pid=102,start=200,members={'102':200})])
        collection=dict(complete=True,code=0,checkout='172181fc5ab4eb2bdc2eb7f37e80d25e482a0887',terminal_processes=[dict(pid=101,start=100),dict(pid=102,start=200)])
        return identity,samples,collection

    def test_resource_evidence(self):
        self.assertEqual(audit.resources(*self.fixture())['samples'],2)
        mutations=[
            lambda j,s,c:j.update(complete=False),lambda j,s,c:j['runs'][0].update(seconds=1800),
            lambda j,s,c:j['runs'][0].update(code=1),lambda j,s,c:j['limits'].update(rss=16*1024**3),
            lambda j,s,c:j['runs'][0].update(samples=1),lambda j,s,c:s[0].update(available_memory=0),
            lambda j,s,c:s[1]['members'][0].update(cpu_seconds=0),lambda j,s,c:s[0]['members'][0].update(affinity='0-3'),
            lambda j,s,c:s[0]['members'][0].update(group=100),lambda j,s,c:s[0]['members'][0].update(start=199),
            lambda j,s,c:c['terminal_processes'].pop(),lambda j,s,c:c.update(checkout='wrong')]
        for index,change in enumerate(mutations):
            with self.subTest(index=index):
                args=self.fixture();change(*args)
                with self.assertRaises(AssertionError):audit.resources(*args)

    def test_prospective_runtime_guards(self):
        sample=self.fixture()[1][0];h.check_sample(sample,102,200)
        for field,value in [('seconds',1800),('available_memory',256*1024**2-1)]:
            changed=copy.deepcopy(sample);changed[field]=value
            with self.assertRaises(AssertionError):h.check_sample(changed,102,200)
        for field,value in [('rss',8*1024**3),('affinity','0'),('group',99),('start',199)]:
            changed=copy.deepcopy(sample);changed['members'][0][field]=value
            with self.assertRaises(AssertionError):h.check_sample(changed,102,200)


if __name__=='__main__':unittest.main()
