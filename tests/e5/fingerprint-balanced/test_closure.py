import copy,unittest
from close_phase import telemetry
from audit import CASES

def fixture():
    runs=[];samples={};clock=100.
    for visit in range(4):
        for index in (range(5) if visit%2==0 else reversed(range(5))):
            name=f'v{visit}-{CASES[index]}';pid=1000+len(runs);birth=clock
            run=dict(job=dict(name=name,case=CASES[index],case_index=index,visit=visit),started=clock,ended=clock+2,seconds=1.5,code=0,
                     samples=2,peak_rss=4096,members={str(pid):birth},child=dict(pid=pid,birth=birth))
            samples[name]=[dict(seconds=t,available=2*1024**3,members=[dict(pid=pid,birth=birth,rss=4096,affinity=[2])]) for t in [.1,1.]]
            runs.append(run);clock+=3
    return dict(started=99.,ended=clock,complete=True,code=0,limits=dict(seconds=300,rss=6*1024**3,available=1024**3),runs=runs),samples

class ClosureTests(unittest.TestCase):
    def test_full_schedule_and_birth_intervals(self):
        state,samples=fixture();self.assertEqual(telemetry(state,samples),40)

    def test_invalid_ownership_resources_and_intervals(self):
        changes=[lambda s,r:s['runs'].reverse(),lambda s,r:s['runs'][1].update(started=99),lambda s,r:s['runs'][0].update(seconds=301),
                 lambda s,r:s['limits'].update(rss=7*1024**3),lambda s,r:r['v0-e5-8tok'][0].update(seconds=2),
                 lambda s,r:r['v0-e5-8tok'][0].update(available=0),lambda s,r:r['v0-e5-8tok'][0]['members'][0].update(affinity=[0]),
                 lambda s,r:r['v0-e5-8tok'][1]['members'][0].update(birth=101),lambda s,r:r['v0-e5-8tok'][0]['members'][0].update(rss=6*1024**3),
                 lambda s,r:r['v0-e5-8tok'][0]['members'].append(r['v0-e5-8tok'][0]['members'][0].copy()),
                 lambda s,r:s['runs'][0].update(samples=3),lambda s,r:s['runs'][0].update(peak_rss=1),lambda s,r:s.update(complete=False)]
        for change in changes:
            state,samples=fixture();change(state,samples)
            with self.assertRaises(AssertionError):telemetry(state,samples)

if __name__=='__main__':unittest.main()
