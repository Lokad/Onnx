"""Check real process liveness and the prospective AMD guard boundaries."""
import copy
import subprocess
import sys
import unittest
import psutil
from supervise import alive
from remote import h
from audit_amd import resources


class ProcessTests(unittest.TestCase):
    def test_birth_and_real_exit_with_open_popen(self):
        child = subprocess.Popen([sys.executable,'-B','-c','import sys; sys.stdin.read()'],stdin=subprocess.PIPE)
        try:
            birth = psutil.Process(child.pid).create_time()
            self.assertTrue(alive(child.pid,birth))
            self.assertFalse(alive(child.pid,birth+1))
        finally:
            child.communicate(timeout=10)
        self.assertEqual(child.returncode,0)
        self.assertFalse(alive(child.pid,birth))

    def fixture(self):
        member = dict(pid=102,start=200,group=102,affinity='2',state='R',rss=4096,cpu_seconds=.1)
        samples = [dict(seconds=t,members=[copy.deepcopy(member)],available_memory=1024**3) for t in (.1,.6)]
        identity = dict(schema=1,complete=True,supervisor=dict(pid=101,start=100,affinity='0'),
            limits=dict(rss=27*1024**3//2,seconds=3600,available_memory=256*1024**2),started=1,ended=4,
            runs=[dict(name='managed',code=0,seconds=1,started=2,ended=3,samples=2,peak_rss=4096,pid=102,start=200,members={'102':200})])
        collection = dict(complete=True,code=0,checkout='172181fc5ab4eb2bdc2eb7f37e80d25e482a0887',
            terminal_processes=[dict(pid=101,start=100),dict(pid=102,start=200)])
        return identity,samples,collection

    def test_amd_resource_evidence(self):
        self.assertEqual(resources(*self.fixture())['samples'],2)
        mutations = [
            lambda j,s,c:j['runs'][0].update(seconds=3600),
            lambda j,s,c:j['runs'][0].update(code=1),
            lambda j,s,c:j['limits'].update(rss=20*1024**3),
            lambda j,s,c:s[0].update(available_memory=0),
            lambda j,s,c:s[1]['members'][0].update(cpu_seconds=0),
            lambda j,s,c:s[0]['members'][0].update(affinity='0-3'),
            lambda j,s,c:s[0]['members'][0].update(group=100),
            lambda j,s,c:c['terminal_processes'].pop(),
        ]
        for index,change in enumerate(mutations):
            with self.subTest(index=index):
                args = self.fixture()
                change(*args)
                with self.assertRaises(AssertionError):
                    resources(*args)

    def test_actual_amd_guards(self):
        sample = self.fixture()[1][0]
        h.check_sample(sample,102,200)
        for field,value in [('seconds',3600),('available_memory',256*1024**2-1)]:
            damaged = copy.deepcopy(sample)
            damaged[field] = value
            with self.assertRaises(AssertionError):h.check_sample(damaged,102,200)
        for field,value in [('rss',27*1024**3//2),('affinity','0'),('group',999),('start',199)]:
            damaged = copy.deepcopy(sample)
            damaged['members'][0][field] = value
            with self.assertRaises(AssertionError):h.check_sample(damaged,102,200)


if __name__ == '__main__':
    unittest.main()
