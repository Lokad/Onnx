from pathlib import Path
from unittest.mock import patch
import copy,unittest
import audit as a

BASE=Path(__file__).resolve().parents[3]/'artifacts/e5-fingerprint-deployment-20260920'

def synthetic(phase):
    values=[]
    for job in a.schedule():
        s=a.specification(job,phase);ticks=80000 if job['role']=='N' else 95000 if phase=='compare' and job['role']=='C' else 100000
        def row(block,call,time):return dict(block=block,call=call,execute=time,request=time+100,bytes=10,g0=0,g1=0,g2=0)
        values.append(dict(specification=s,frequency=1000000,first=row(-1,0,ticks),conditioning=[row(-2,0,30000000)],conditioned=30.,conditioning_wall_ticks=30001000,
                           measured=[row(b,c,ticks) for b in range(s['blocks']) for c in range(s['calls'])]))
    return values

class AuditTests(unittest.TestCase):
    def test_full_schedule_and_counts(self):
        jobs=a.schedule();self.assertEqual(len(jobs),160);self.assertEqual(len({j['name'] for j in jobs}),160)
        for index in range(5):
            for policy in a.POLICIES:
                group=[j for j in jobs if j['case_index']==index and j['policy']==policy]
                for role in a.ROLES:self.assertEqual(sorted(j['position'] for j in group if j['role']==role),list(range(4)))
                for visit in range(4):self.assertEqual(sorted(j['role'] for j in group if j['visit']==visit),sorted(a.ROLES))
        self.assertEqual(sum(a.specification(j,'aa')['blocks']*a.specification(j,'aa')['calls'] for j in jobs),89088)

    def test_identical_controls_and_known_gain(self):
        for phase in ['aa','compare']:
            result=a.evaluate(synthetic(phase),phase);self.assertTrue(result['passed']);self.assertEqual(len(result['cases']),10)
            self.assertAlmostEqual(result['cases'][0]['boundaries']['execute']['native_ratios']['A'],1.25)

    def test_reject_control_candidate_and_position_bias(self):
        cases=[('aa',lambda j:j['role']=='B',1.02),('aa',lambda j:j['role']=='C' and j['position']==0,1.03),
               ('compare',lambda j:j['role']=='C',1.05),('compare',lambda j:j['role']=='C' and j['visit']==3 and j['case_index']==4,1.1)]
        for phase,selected,factor in cases:
            values=synthetic(phase)
            for j,v in zip(a.schedule(),values):
                if selected(j):
                    for r in v['measured']:r['execute']=round(r['execute']*factor);r['request']=round(r['request']*factor)
            self.assertFalse(a.evaluate(values,phase)['passed'])

    def test_retain_every_gc_tail_and_both_boundaries(self):
        values=synthetic('aa');before=a.evaluate(values,'aa')['cases'][0]['boundaries']['execute']['role_mean_seconds']['A']
        target=next(v for v in values if v['specification']['policy']=='default' and v['specification']['case_index']==0 and v['specification']['role']=='A')
        target['measured'][0]['execute']+=1000000;target['measured'][0]['request']+=1000000;target['measured'][0]['g0']=1
        after=a.evaluate(values,'aa')['cases'][0]['boundaries']['execute']['role_mean_seconds']['A']
        self.assertAlmostEqual(after-before,1/(4*48*32),places=14)
        value=synthetic('compare')
        for v in value:
            if v['specification']['role']=='C':
                for r in v['measured']:r['request']+=10000
        self.assertFalse(a.evaluate(value,'compare')['passed'])

    def test_malformed_sampling_refused(self):
        value=synthetic('aa')[0]
        for mutate in [lambda v:v['measured'].pop(),lambda v:v['measured'].reverse(),lambda v:v.update(frequency=0),
                       lambda v:v['measured'][0].update(execute=0),lambda v:v['measured'][0].update(request=1),
                       lambda v:v['measured'][0].update(g2=-1),lambda v:v.update(conditioned=29.),lambda v:v['conditioning'].append(v['conditioning'][0].copy())]:
            v=copy.deepcopy(value);mutate(v)
            with self.assertRaises(AssertionError):a.rows(v)

    def test_all_actual_smokes_and_damaged_records(self):
        meta=a.read(BASE/'smoke-audit.json');self.assertTrue(meta['passed']);self.assertEqual(len(meta['jobs']),13)
        for job in meta['jobs']:
            a.worker(BASE/'smoke-process'/job['name']/'output',BASE/'inputs',meta['model'],meta['binaries']['FingerprintDeployment.dll'],meta['binaries']['Microsoft.ML.OnnxRuntime.dll'],meta['native'],job,job['phase'],True)
        job=meta['jobs'][0];folder=BASE/'smoke-process'/job['name']/'output';original=a.read(folder/'result.json');read=a.read
        mutations=[lambda v:v.update(passed=False),lambda v:v.update(core_sha256='0'*64),lambda v:v.update(enabled=True),lambda v:v.update(cache_entries=2330),
                   lambda v:v.update(cache_sha256='0'*64),lambda v:v.update(native_identity={}),lambda v:v.update(affinity=1),lambda v:v.update(flags={'DOTNET_JitOSR':'0'}),
                   lambda v:v.update(unchanged_inputs=False),lambda v:v.update(unchanged_held_output=False),lambda v:v.update(unchanged_cache=False),
                   lambda v:v.update(output_sha256='0'*64),lambda v:v.update(shape=[1,7,384]),lambda v:v.update(after_error=1),lambda v:v.update(optimization='Memory'),
                   lambda v:v.update(ort_managed_sha256='0'*64),lambda v:v['specification'].update(role='C')]
        for change in mutations:
            value=copy.deepcopy(original);change(value)
            with patch.object(a,'read',side_effect=lambda p:value if p==folder/'result.json' else read(p)):
                with self.assertRaises(AssertionError):a.worker(folder,BASE/'inputs',meta['model'],meta['binaries']['FingerprintDeployment.dll'],meta['binaries']['Microsoft.ML.OnnxRuntime.dll'],meta['native'],job,job['phase'],True)
        job=next(j for j in meta['jobs'] if j['role']=='N');folder=BASE/'smoke-process'/job['name']/'output';value=read(folder/'result.json');value['native_identity']['sha256']='0'*64
        with patch.object(a,'read',side_effect=lambda p:value if p==folder/'result.json' else read(p)):
            with self.assertRaises(AssertionError):a.worker(folder,BASE/'inputs',meta['model'],meta['binaries']['FingerprintDeployment.dll'],meta['binaries']['Microsoft.ML.OnnxRuntime.dll'],meta['native'],job,job['phase'],True)

    def test_actual_resources_and_refusals(self):
        meta=a.read(BASE/'smoke-audit.json');state=a.read(BASE/'smoke-process/identity.json')
        samples={j['name']:[a.json.loads(line) for line in (BASE/'smoke-process'/j['name']/'samples.jsonl').read_text().splitlines()] for j in meta['jobs']}
        self.assertEqual(a.telemetry(state,samples,meta['jobs']),meta['resources'])
        first=meta['jobs'][0]['name'];index=next(i for i,r in enumerate(samples[first]) if r['members'])
        for change in [lambda s,r:s.update(code=1),lambda s,r:s['runs'].reverse(),lambda s,r:s['runs'][0].update(peak_rss=0),
                       lambda s,r:s['runs'][0].update(samples=0),lambda s,r:s['runs'][0]['child'].update(birth=0),
                       lambda s,r:r[first][0].update(available=0),lambda s,r:r[first][0].update(seconds=301),
                       lambda s,r:r[first][index]['members'][0].update(affinity=[0]),
                       lambda s,r:r[first][index]['members'].append(r[first][index]['members'][0].copy())]:
            s=copy.deepcopy(state);r=copy.deepcopy(samples);change(s,r)
            with self.assertRaises(AssertionError):a.telemetry(s,r,meta['jobs'])

    def test_report_includes_each_policy_boundary_and_native(self):
        from close_phase import report
        for phase in ['aa','compare']:
            result=a.evaluate(synthetic(phase),phase);text=report(dict(timing=result))
            self.assertEqual(len(text.splitlines()),20)
            for case in a.CASES:
                for policy in a.POLICIES:
                    for boundary in ['execute','request']:self.assertIn(f'| {case} | {policy} | {boundary} |',text)
            self.assertIn('| 80.0000 |',text)

    def test_native_settings_and_nonfinite_arrays_refused(self):
        meta=a.read(BASE/'smoke-audit.json');job=next(j for j in meta['jobs'] if j['role']=='N');folder=BASE/'smoke-process'/job['name']/'output'
        good=a.read(folder/'result.json');read=a.read
        for key,value in [('version','0'),('threads',2),('sequential',False),('all_optimizations',False),('spinning',True)]:
            v=copy.deepcopy(good);v['native_identity'][key]=value
            with patch.object(a,'read',side_effect=lambda p:v if p==folder/'result.json' else read(p)):
                with self.assertRaises(AssertionError):a.worker(folder,BASE/'inputs',meta['model'],meta['binaries']['FingerprintDeployment.dll'],meta['binaries']['Microsoft.ML.OnnxRuntime.dll'],meta['native'],job,job['phase'],True)
        for values in [a.np.array([1.],dtype='<f4'),a.np.full(8*384,a.np.nan,dtype='<f4'),a.np.full(8*384,a.np.inf,dtype='<f4')]:
            with patch.object(a.np,'fromfile',return_value=values):
                with self.assertRaises(AssertionError):a.worker(folder,BASE/'inputs',meta['model'],meta['binaries']['FingerprintDeployment.dll'],meta['binaries']['Microsoft.ML.OnnxRuntime.dll'],meta['native'],job,job['phase'],True)

if __name__=='__main__':unittest.main()
