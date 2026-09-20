from pathlib import Path
from unittest.mock import patch
import copy,itertools,unittest
import audit as a

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/e5-fingerprint-balanced-20260920'

def synthetic(phase):
    result=[]
    for index,visit in itertools.product(range(5),range(4)):
        s=a.schedule(phase,visit,index);rows=[]
        for cycle,order in enumerate(s['order']):
            for position,role in enumerate(a.PERMUTATIONS[order]):
                for call in range(s['calls']):
                    ticks=100000
                    if phase=='compare' and role==2:ticks=95000
                    rows.append(dict(role=role,stage='measured',cycle=cycle,position=position,call=call,
                        enabled=phase=='compare' and role==2,execute=ticks,request=ticks+100,bytes=100,g0=0,g1=0,g2=0))
        result.append(dict(schedule=s,frequency=1000000,measured=rows))
    return result

class EvidenceTests(unittest.TestCase):
    def test_every_schedule_is_balanced(self):
        for phase,visit,index in itertools.product(['aa','compare'],range(4),range(5)):
            s=a.schedule(phase,visit,index)
            self.assertEqual(sorted(s['order']),sorted(list(range(6))*8))
            for position in range(3):
                self.assertEqual(sorted(a.PERMUTATIONS[p][position] for p in s['order']),sorted([0,1,2]*16))
            for offset in range(0,48,6):
                block=s['order'][offset:offset+6]
                self.assertEqual(sorted(block),list(range(6)))
                for position in range(3):
                    self.assertEqual(sorted(a.PERMUTATIONS[p][position] for p in block),[0,0,1,1,2,2])

    def test_block_balance_rejects_globally_balanced_order(self):
        # A global permutation census alone would accept this clustered order.
        old=sorted(list(range(6))*8)
        self.assertEqual(sorted(old),sorted(a.schedule('aa',0,0)['order']))
        self.assertNotEqual(old,a.schedule('aa',0,0)['order'])
        self.assertNotEqual(sorted(old[:6]),list(range(6)))

    def test_piecewise_constant_drift_cancels_within_each_block(self):
        values=synthetic('aa')
        for v in values:
            for row in v['measured']:
                factor=1+.15*(row['cycle']//6)+.04*row['position']
                row['execute']=round(row['execute']*factor)
                row['request']=row['execute']+100
        self.assertTrue(a.evaluate(values,'aa')['passed'])

    def test_equal_controls_and_fixed_candidate_gain_pass(self):
        for phase in ['aa','compare']:self.assertTrue(a.evaluate(synthetic(phase),phase)['passed'])

    def test_control_candidate_and_order_bias_are_refused(self):
        changes=[('aa',lambda v:[r.update(execute=r['execute']*1.02,request=r['request']*1.02) for r in v[0]['measured'] if r['role']==1]),
                 ('aa',lambda v:[r.update(execute=r['execute']*1.03,request=r['request']*1.03) for item in v for r in item['measured'] if r['role']==2 and r['position']==0]),
                 ('compare',lambda v:[r.update(execute=99500,request=99600) for item in v for r in item['measured'] if r['role']==2]),
                 ('compare',lambda v:[r.update(execute=105000,request=105100) for r in v[-1]['measured'] if r['role']==2])]
        for phase,change in changes:
            value=synthetic(phase);change(value);self.assertFalse(a.evaluate(value,phase)['passed'])

    def test_every_tail_sample_contributes(self):
        value=synthetic('aa');before=a.evaluate(value,'aa')['cases'][0]['boundaries']['execute']['role_mean_seconds'][0]
        row=next(r for r in value[0]['measured'] if r['role']==0);row['execute']+=1000000;row['request']+=1000000;row['g0']=1
        after=a.evaluate(value,'aa')['cases'][0]['boundaries']['execute']['role_mean_seconds'][0]
        n=sum(r['role']==0 for v in value if v['schedule']['case_index']==0 for r in v['measured'])
        self.assertAlmostEqual(after-before,1/n,places=14)

    def test_actual_smoke_outputs_and_mutations(self):
        output=BASE/'smoke-process/output';meta=a.read(BASE/'smoke-audit.json');good=a.read(output/'result.json')
        result=a.worker(output,BASE/'inputs',meta['model'],meta['binaries']['FingerprintBalanced.dll'],'smoke',0,0)
        self.assertEqual(len(result['measured']),72)
        mutations=[lambda v:v.update(passed=False),lambda v:v.update(core_sha256='0'*64),lambda v:v.update(affinity=1),
                   lambda v:v.update(unchanged_inputs=False),lambda v:v.update(unchanged_held_output=False),lambda v:v.update(unchanged_cache=False),
                   lambda v:v.update(frequency=0),lambda v:v['measured'].pop(),lambda v:v['measured'].reverse(),
                   lambda v:v['measured'][0].update(enabled=not v['measured'][0]['enabled']),lambda v:v['measured'][0].update(execute=-1),
                   lambda v:v['measured'][0].update(request=0),lambda v:v['measured'][0].update(g1=-1),
                   lambda v:v.update(output_sha256='0'*64),lambda v:v.update(after_error=1.),lambda v:v.update(shape=[1,7,384]),
                   lambda v:v['flags'].update(DOTNET_JitOSR='0'),lambda v:v['schedule']['order'].reverse()]
        read=a.read
        for mutate in mutations:
            value=copy.deepcopy(good);mutate(value)
            with patch.object(a,'read',side_effect=lambda p:value if p==output/'result.json' else read(p)):
                with self.assertRaises((AssertionError,IndexError)):a.worker(output,BASE/'inputs',meta['model'],meta['binaries']['FingerprintBalanced.dll'],'smoke',0,0)

if __name__=='__main__':unittest.main()
