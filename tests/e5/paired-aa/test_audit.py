import copy,json,shutil,tempfile,unittest
from pathlib import Path
from audit import schedule,measured_order,rows_valid,timing,outputs,resources
from prepare_inputs import CASES,read

def fake(visit,index):
    s=schedule(visit,index);frequency=1000
    def row(arm,phase,group,position,ticks):return dict(arm=arm,phase=phase,group=group,position=position,execute=ticks,request=ticks+1,bytes=100,g0=0,g1=0,g2=0)
    conditioning=[row(a,'conditioning',0,p,30000) for p,a in enumerate(s['creation'])]
    measured=[row(a,p,g,i,1000+(g%7)*10) for a,p,g,i in measured_order(s)]
    return dict(schedule=s,frequency=frequency,conditioning=conditioning,conditioned=[30.,30.],measured=measured)

class AuditTests(unittest.TestCase):
    def test_balanced_schedule_and_common_drift(self):
        records=[fake(v,i) for v in range(4) for i in range(5)]
        for value in records:
            rows_valid(value,False)
            self.assertEqual(value['schedule']['pair_first'].count(0),32)
        self.assertTrue(timing(records)['passed'])

    def test_known_bias_order_and_coupling_are_rejected(self):
        for kind in ('arm','order','solo','gc_tail'):
            records=[fake(v,i) for v in range(4) for i in range(5)]
            for value in records:
                for row in value['measured']:
                    if kind=='arm' and row['arm']==1:row['execute']*=1.02;row['request']*=1.02
                    if kind=='order' and row['position']==0 and row['phase']=='paired':row['execute']*=1.03;row['request']*=1.03
                    if kind=='solo' and row['phase']=='solo':row['execute']*=.9;row['request']*=.9
                    if kind=='gc_tail' and row['arm']==1 and row['phase']=='paired' and row['group']==0:row['execute']*=2;row['request']*=2
            self.assertFalse(timing(records)['passed'],kind)

    def test_damaged_actual_schedule_is_rejected(self):
        original=fake(0,0)
        mutations=[lambda v:v['measured'].pop(),lambda v:v['measured'].reverse(),
            lambda v:v['measured'][0].update(arm=1-v['measured'][0]['arm']),
            lambda v:v['measured'][0].update(request=0),lambda v:v['conditioning'].pop(),
            lambda v:v['schedule'].update(conditioning_seconds=0),lambda v:v.update(conditioned=[1.,1.])]
        for mutate in mutations:
            v=copy.deepcopy(original);mutate(v)
            with self.assertRaises((AssertionError,IndexError)):rows_valid(v,False)

    def test_real_smoke_full_arrays(self):
        base=Path(__file__).resolve().parents[3]/'artifacts/e5-paired-aa-v2-20260920'
        value,error=outputs(base/'smoke',base/'inputs',read(base/'smoke-process.json')['binaries'],True)
        self.assertLess(error,1e-4);self.assertEqual(len(value['measured']),16)

    def test_damaged_real_smoke(self):
        base=Path(__file__).resolve().parents[3]/'artifacts/e5-paired-aa-v2-20260920'
        mutations=[lambda v:v.update(static_isolation=False),lambda v:v.update(private_cores=False),
            lambda v:v['initial'][0].update(core_sha256='0'*64),lambda v:v['final'][1].update(held_outputs_unchanged=False),
            lambda v:v['initial'][0].update(shape=[1,7,384]),lambda v:v['flags'][1].update(EnableBiasGeluInterleaved=True),
            lambda v:v['initial'][0]['inputs']['input_ids'].__setitem__(0,7),lambda v:v['measured'].pop()]
        with tempfile.TemporaryDirectory() as temporary:
            for index,mutate in enumerate(mutations):
                target=Path(temporary)/str(index);shutil.copytree(base/'smoke',target)
                v=read(target/'result.json');mutate(v);(target/'result.json').write_text(json.dumps(v),encoding='utf-8')
                with self.assertRaises((AssertionError,IndexError)):
                    outputs(target,base/'inputs',read(base/'smoke-process.json')['binaries'],True)
            target=Path(temporary)/'array';shutil.copytree(base/'smoke',target)
            (target/'arm0/after.f32').write_bytes(b'\0'*12)
            with self.assertRaises(AssertionError):outputs(target,base/'inputs',read(base/'smoke-process.json')['binaries'],True)

    def test_process_evidence_and_resource_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);(base/'result').mkdir()
            identity=dict(complete=True,supervisor=dict(pid=1,start=1,affinity='0'),started=0,ended=100,runs=[],
                limits=dict(rss=8*1024**3,seconds=600,available_memory=1024**3))
            collection=dict(complete=True,code=0,checkout='172181fc5ab4eb2bdc2eb7f37e80d25e482a0887',terminal_processes=[dict(pid=1,start=1)])
            sample_paths=[]
            for index,(v,i) in enumerate((v,i) for v in range(4) for i in (range(5) if v%2==0 else reversed(range(5)))):
                pid=index+2;row=dict(name=f'v{v}-{CASES[i]}',pid=pid,start=pid,started=index*4,ended=index*4+3,seconds=3,
                    code=0,samples=2,peak_rss=1000,members={str(pid):pid},accounting=dict(valid=True,foreign_cpu_fraction=0))
                identity['runs'].append(row);collection['terminal_processes'].append(dict(pid=pid,start=pid))
                samples=[dict(seconds=t,available_memory=2*1024**3,members=[dict(pid=pid,start=pid,group=pid,affinity='2',state='R',rss=1000,cpu_seconds=t)]) for t in (.5,1.)]
                path=base/'result'/(row['name']+'-samples.jsonl');path.write_text('\n'.join(json.dumps(s) for s in samples));sample_paths.append(path)
            resources(identity,collection,base)
            mutations=[lambda j:j.update(complete=False),lambda j:j['runs'][0].update(code=1),
                lambda j:j['runs'][0].update(samples=3),lambda j:j['runs'][0].update(start=99),
                lambda j:j['runs'][0].update(peak_rss=1),lambda j:j['limits'].update(seconds=601),
                lambda j:j['runs'].reverse()]
            for mutate in mutations:
                j=copy.deepcopy(identity);mutate(j)
                with self.assertRaises(AssertionError):resources(j,collection,base)
            path=sample_paths[0];original=path.read_text()
            for mutate in (lambda s:s.update(available_memory=0),lambda s:s.update(seconds=5),
                lambda s:s['members'][0].update(rss=8*1024**3),lambda s:s['members'][0].update(affinity='0'),
                lambda s:s['members'][0].update(state='Z')):
                samples=[json.loads(line) for line in original.splitlines()];mutate(samples[0]);path.write_text('\n'.join(json.dumps(s) for s in samples))
                with self.assertRaises(AssertionError):resources(identity,collection,base)
            path.write_text(original)

    def test_actual_runtime_guard(self):
        from remote import h
        valid=dict(seconds=1,available_memory=2*1024**3,members=[dict(pid=2,start=2,group=2,affinity='2',rss=1000,cpu_seconds=1)])
        h.check_sample(valid,2,2)
        for mutate in (lambda s:s.update(seconds=600),lambda s:s.update(available_memory=0),
            lambda s:s['members'][0].update(rss=8*1024**3),lambda s:s['members'][0].update(affinity='0'),
            lambda s:s['members'][0].update(start=3),lambda s:s['members'].append(s['members'][0].copy())):
            s=copy.deepcopy(valid);mutate(s)
            with self.assertRaises(AssertionError):h.check_sample(s,2,2)

if __name__=='__main__':unittest.main()
