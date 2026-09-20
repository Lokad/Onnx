import copy
import unittest
from audit import CASES,REPEATS,VARIANTS,inspect_timing,summarize

def fixture():
    value=dict(schema=1,protocol='uniform-gelu-bank-v1',visit=0,runtime='.NET 10.0.8',affinity=4,avx512=True,flags=[],frequency=1000000000,case_order=CASES,results=[])
    for i,name in enumerate(CASES):
        row=dict(name=name,layers=12,values=[147456,552960,2359296,2359296,9437184][i],held_unchanged=True,output_sha256='f'*64)
        for label,count in [('warmup',16),('measured',48)]:
            row[label]=[dict(cycle=c,position=p,variant=VARIANTS[(c+p)%4],repeats=REPEATS[i],ticks=1000000,allocated=0,gc=[0,0,0],output_sha256='f'*64) for c in range(count) for p in range(4)]
        value['results'].append(row)
    return value

class AuditTests(unittest.TestCase):
    def test_real_protocol_coverage_and_corrupt_sample_refusals(self):
        value=fixture();self.assertEqual(len(inspect_timing(value,0)),20)
        mutations=[lambda v:v['results'].pop(),lambda v:v['results'][0]['measured'].pop(),lambda v:v['results'][0]['measured'][0].update(repeats=63),lambda v:v['results'][0]['measured'][0].update(variant='Conditional'),lambda v:v['results'][0]['measured'][0].update(output_sha256='0'*64),lambda v:v['results'][0]['measured'][0].update(ticks=0),lambda v:v.update(flags=['DOTNET_TieredCompilation']),lambda v:v.update(case_order=list(reversed(CASES)))]
        for change in mutations:
            damaged=copy.deepcopy(value);change(damaged)
            with self.assertRaises(AssertionError):inspect_timing(damaged,0)
    def test_screen_does_not_hide_failed_controls_or_one_failed_case(self):
        rows=[dict(case=c,visit=i,variant=v,mean_ms=.95 if v=='Conditional' else 1.) for c in CASES for i in range(4) for v in VARIANTS]
        self.assertTrue(summarize(rows)['overall_passed'])
        changed=copy.deepcopy(rows)
        for r in changed:
            if r['variant']=='CopyB':r['mean_ms']=1.03
        self.assertFalse(summarize(changed)['controls_passed'])
        changed=copy.deepcopy(rows)
        for r in changed:
            if r['case']=='e5-128tok' and r['variant']=='Conditional':r['mean_ms']=.99
        self.assertFalse(summarize(changed)['candidate_screen_passed'])
        changed=copy.deepcopy(rows)
        for r in changed:
            if r['case']=='e5-512tok' and r['variant']=='Conditional' and r['visit']==0:r['mean_ms']=1.03
        self.assertFalse(summarize(changed)['candidate_screen_passed'])

if __name__=='__main__':unittest.main()
