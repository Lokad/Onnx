"""Ensure incomplete/altered captures and unreliable or regressing timings cannot admit."""
import copy
from pathlib import Path
import unittest
from checks import check_events,score,schedule
from protocol import read

ROOT=Path(__file__).resolve().parents[3]
CAPTURE=read(ROOT/'artifacts/pyannote-lstm-input-blocks-amd-v2-20260922/collected/references/capture.json')


def events():
    rows=[];tick=1
    for ordinal in range(12):
        rows.append(dict(kind='prepare',ordinal=ordinal,start=tick,end=tick+1,frequency=1000));tick+=2
    for p in [-1,0,1,2]:
        for c,source in zip(schedule(CAPTURE)['cases'],CAPTURE['calls']):
            for r in range(c['repeats']):
                rows.append(dict(kind='call',ordinal=c['ordinal'],pass_index=p,repeat=r,start=tick,end=tick+10,frequency=1000,ok=True));rows[-1]['pass']=rows[-1].pop('pass_index');tick+=11
                rows.append(dict(kind='verified',ordinal=c['ordinal'],repeat=r,hashes=[v['sha256'] for v in source['outputs']],
                    errors=[1e-6]*3,values=[v['values'] for v in source['outputs']],
                    scratch=source['inputs'][1]['bytes']+source['inputs'][2]['bytes']+8192,readonly_operands=True,held_outputs_unchanged=True))
                rows[-1]['pass']=p
    return rows


class Check(unittest.TestCase):
    def test_complete_and_mutations(self):
        original=events();value=check_events(original,'time','candidate',CAPTURE,1000)
        self.assertEqual(value['clocks'],588);self.assertEqual(value['warmups'],147)
        def reject(action):
            rows=copy.deepcopy(original);action(rows)
            with self.assertRaises(AssertionError):check_events(rows,'time','candidate',CAPTURE,1000)
        reject(lambda v:v.pop())
        reject(lambda v:v[12].update(repeat=1))
        reject(lambda v:v[12].update(end=v[12]['start']))
        reject(lambda v:v[12].update(frequency=1))
        reject(lambda v:v[12].update(ok=False))
        reject(lambda v:v[13]['hashes'].__setitem__(2,'0'*64))
        reject(lambda v:v[13]['errors'].__setitem__(0,2e-4))
        reject(lambda v:v[13].update(scratch=0))
        reject(lambda v:v[13].update(held_outputs_unchanged=False))
        reject(lambda v:v[13].update(readonly_operands=False))
    def test_gates(self):
        rows={name:dict(means=[1. if name.startswith('selected') else .88]*12) for name in ['selected-0','candidate-1','candidate-2','selected-3']}
        self.assertTrue(score(rows)['admitted'])
        bad=copy.deepcopy(rows);bad['selected-3']['means']=[1.11]*12
        self.assertFalse(score(bad)['admitted'])
        bad=copy.deepcopy(rows)
        for n in ['candidate-1','candidate-2']:bad[n]['means']=[.91]*12
        self.assertFalse(score(bad)['admitted'])
        bad=copy.deepcopy(rows)
        for n in ['candidate-1','candidate-2']:bad[n]['means']=[1.06 if i%4==0 else .8 for i in range(12)]
        result=score(bad);self.assertTrue(result['gates'][0]['passed']);self.assertFalse(result['admitted'])
        bad=copy.deepcopy(rows);bad['candidate-2']['means'][0]=float('nan')
        with self.assertRaises(AssertionError):score(bad)


if __name__=='__main__':unittest.main()
