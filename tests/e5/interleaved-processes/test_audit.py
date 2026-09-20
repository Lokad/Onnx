"""Refuse damaged real smoke evidence and synthetic timing/scheduling biases."""
from pathlib import Path
from unittest.mock import patch
import argparse,copy,itertools,json,sys,unittest
import audit
from protocol import schedule,cycles,commands,specification,ROLES,LIMITS

BASE=None

def synthetic(phase='aa'):
    result=[]
    for job in schedule():
        orders=cycles(job)
        for role in ROLES:
            spec=specification(job|dict(role=role),phase)
            values=[]
            for block in range(48):
                for call in range(spec['calls']):
                    # Shared smooth drift is retained, with every local position.
                    duration=100000+100*block
                    values.append(dict(block=block,call=call,execute=duration,request=duration+100,bytes=100,g0=0,g1=0,g2=0))
            stage=0 if role==job['creation'][0] else 1 if role==job['creation'][-1] else -1
            result.append(dict(specification=spec,frequency=1000000,measured=values,
                solo=[dict(execute=102350,request=102450,bytes=100,g0=0,g1=0,g2=0) for _ in range(64)] if stage>=0 else [],solo_stage=stage))
    return result

class AuditTests(unittest.TestCase):
    def test_report_covers_all_boundaries_and_bridge_failures(self):
        from close_phase import report
        value=dict(phase='aa',timing=audit.evaluate(synthetic(),'aa'),measured_calls=89088,conditioning_calls=12345,
            solo_calls=5120,maximum_native_error=1e-6,resources=dict(cohorts=[dict(samples=10)]))
        text=report(value,dict(source_revision='test'))
        self.assertIn('passes',text);self.assertEqual(20,sum(line.startswith('| e5-') for line in text.splitlines()))
        value['timing']['passed']=False;value['timing']['cases'][0]['boundaries']['execute']['bridges_passed']=False
        self.assertIn('solo/resident',report(value,dict(source_revision='test')))
        self.assertIn('candidate phase is not run',report(value,dict(source_revision='test')))

    def test_schedule_balance(self):
        jobs=schedule();self.assertEqual(40,len(jobs))
        for j in jobs:
            orders=cycles(j);self.assertEqual(48,len(orders))
            for start in [0,24]:self.assertEqual(set(itertools.permutations(ROLES)),{tuple(v) for v in orders[start:start+24]})
            self.assertEqual(206,len(commands(j)))
        for case in range(5):
            for policy in ['default','memory']:
                group=[j for j in jobs if j['case_index']==case and j['policy']==policy]
                self.assertEqual(set(ROLES),{j['creation'][0] for j in group})
                self.assertEqual(set(ROLES),{j['creation'][-1] for j in group})

    def test_balanced_drift_and_declared_candidate(self):
        self.assertTrue(audit.evaluate(synthetic(),'aa')['passed'])
        values=synthetic('compare')
        for v in values:
            if v['specification']['role']=='C':
                for row in v['measured']+v['solo']:
                    for boundary in ['execute','request']:row[boundary]=int(row[boundary]*.97)
        self.assertTrue(audit.evaluate(values,'compare')['passed'])

    def test_bias_and_gc_tail_never_discarded(self):
        for label in ['role','visit','position','coupling','gc-tail']:
            values=synthetic()
            for v in values:
                s=v['specification']
                if s['case_index']!=0 or s['policy']!='memory' or s['role']!='B':continue
                if label=='coupling':
                    for r in v['solo']:r['execute']*=2;r['request']*=2
                else:
                    orders=cycles(next(j for j in schedule() if j['visit']==s['visit'] and j['case_index']==0 and j['policy']=='memory'))
                    for r in v['measured']:
                        affected=label=='role' or label=='visit' and s['visit']==0 or label=='position' and orders[r['block']][0]=='B' or label=='gc-tail' and r['block']==0 and r['call']==0
                        if affected:
                            factor=100 if label=='gc-tail' else 1.08
                            r['execute']=int(r['execute']*factor);r['request']=int(r['request']*factor);r['g0']=1
            self.assertFalse(audit.evaluate(values,'aa')['passed'],label)

    def test_real_worker_record_refusals(self):
        meta=audit.read(BASE/'smoke-audit.json');job=next(j for j in schedule() if j['visit']==0 and j['case_index']==0 and j['policy']=='memory')
        target=BASE/'smoke-compare'/job['name']/'C/output/result.json';good=audit.read(target);real=audit.read
        def check():return audit.worker(target.parent,BASE/'inputs',meta['model'],meta['binaries']['InterleavedProcesses.dll'],
            meta['binaries']['Microsoft.ML.OnnxRuntime.dll'],meta['native'],job|dict(role='C'),'compare',True)
        check()
        changes=[('done',lambda v:v.update(passed=False)),('wide-switch',lambda v:v.update(wide_enabled=False)),
            ('cache-switch',lambda v:v.update(enabled=False)),('flags',lambda v:v['flags'].update(DOTNET_JitOSR='0')),
            ('core',lambda v:v.update(core_sha256='0'*64)),('output',lambda v:v.update(output_sha256='0'*64)),
            ('shape',lambda v:v['shape'].__setitem__(1,7)),('input',lambda v:v.update(input_sha256='0'*64)),
            ('held',lambda v:v.update(unchanged_held_output=False)),('cache',lambda v:v.update(cache_entries=0)),
            ('missing-call',lambda v:v['measured'].pop()),('duplicate-call',lambda v:v['measured'].append(v['measured'][0])),
            ('negative-time',lambda v:v['measured'][0].update(execute=-1)),('enclosing-time',lambda v:v['measured'][0].update(request=1)),
            ('batch-order',lambda v:v['measured'][0].update(block=1)),('solo-stage',lambda v:v.update(solo_stage=0)),
            ('conditioning-minimum',lambda v:v['conditioning'].clear())]
        for label,change in changes:
            damaged=copy.deepcopy(good);change(damaged)
            with patch.object(audit,'read',side_effect=lambda path:damaged if path==target else real(path)):
                with self.assertRaises(AssertionError,msg=label):check()

    def test_real_command_and_resource_refusals(self):
        folder=BASE/'smoke-aa';good=audit.read(folder/'identity.json');jobs=[j for j in schedule() if j['visit']==0]
        records={j['name']:[json.loads(l) for l in (folder/j['name']/'samples.jsonl').read_text().splitlines()] for j in jobs}
        audit.telemetry(good,records,jobs,True)
        changes=[('incomplete',lambda s:s.update(complete=False)),('missing-command',lambda s:s['runs'][0]['events'].pop()),
            ('wrong-ack',lambda s:s['runs'][0]['events'][0].update(ack=[3,0])),('wrong-role',lambda s:s['runs'][0]['events'][0].update(role='N')),
            ('birth',lambda s:s['runs'][0]['workers']['A'].update(birth=0)),('deadline',lambda s:s['runs'][0].update(seconds=601)),
            ('rss-total',lambda s:s['runs'][0].update(peak_rss=1)),('samples',lambda s:s['runs'][0].update(samples=0))]
        for label,change in changes:
            damaged=copy.deepcopy(good);change(damaged)
            with self.assertRaises(AssertionError,msg=label):audit.telemetry(damaged,records,jobs,True)
        name=jobs[0]['name']
        for label in ['memory','rss','affinity','overlap','birth']:
            damaged=copy.deepcopy(records)
            row=next(r for r in damaged[name] if r['members'] and any(m['role']!=r['active'] for m in r['members']))
            member=next(m for m in row['members'] if m['role']!=row['active'])
            if label=='memory':row['available']=0
            elif label=='rss':member['rss']=LIMITS['rss']
            elif label=='affinity':member['affinity']=[0]
            elif label=='overlap':member['suspended']=False
            else:member['birth']=0
            with self.assertRaises(AssertionError,msg=label):audit.telemetry(good,damaged,jobs,True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a,rest=p.parse_known_args();BASE=a.artifact.resolve()
    unittest.main(argv=[sys.argv[0]]+rest)
