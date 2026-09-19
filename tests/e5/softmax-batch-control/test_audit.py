import copy,hashlib,struct,unittest
from audit import CORE,ITERATIONS,MODES,SHAPES,WIDTHS,evaluate,sequence,validate_worker

TELEMETRY=dict(maximum_foreign_cpu_fraction=0.,maximum_steal_fraction=0.)

def worker(order,phase='control'):
    records=[]
    for i,name in enumerate(SHAPES):
        columns=WIDTHS[i];active=30 if name.startswith('pad') else columns
        mask=struct.pack('<f',0)*active+struct.pack('<I',0xff7fffff)*(columns-active)
        for mode in MODES:
            candidate=phase=='compare' and mode=='probe'
            implementation='ZeroBlocks.Kernels.Adaptive' if candidate else 'ZeroBlocks.Kernels.Original'
            if mode=='actual':implementation='Lokad.Onnx.Tensor`1[System.Single].SoftmaxMaskedFloatSpanPtr'
            ticks=80_000_000 if candidate and name.startswith('pad') else 100_000_000
            records.append(dict(name=name,mode=mode,implementation=implementation,rows=12*columns,columns=columns,iterations=ITERATIONS[i],conditioning_calls=ITERATIONS[i]*20,
                input_sha256='input-'+name,mask_sha256=hashlib.sha256(mask).hexdigest(),output_sha256='output-'+name,
                samples=[dict(ticks=ticks,thread_ns=ticks,process_ns=ticks,gc=[0,0,0]) for _ in range(9)]))
    return dict(schema=2,protocol='softmax-long-batches-v1',phase=phase,order=order,sequence=sequence(order),checkOnly=False,width=8,frequency=1_000_000_000,runtime='.NET 10.0.8',
        core_sha256=CORE,expValues=2503712,tensorCases=918,refusals=4,maximumDoubleError=1e-7,records=records)

def workers(phase='control'):return [worker(i,phase) for i in range(8)]
def row(w,name,mode):return next(r for r in w['records'] if r['name']==name and r['mode']==mode)
def change(w,name,mode,ticks):
    for s in row(w,name,mode)['samples']:s['ticks']=ticks

class Audit(unittest.TestCase):
    def test_passing_control_and_compare(self):
        for phase in ('control','compare'):self.assertTrue(evaluate(workers(phase),phase,TELEMETRY)['passed'])
    def test_bad_identity_and_schedule(self):
        for key,value in [('schema',1),('protocol','old'),('phase','compare'),('order',1),('sequence',[0,2,1,3]),('checkOnly',True),('width',16),('frequency',0),('runtime','wrong'),('core_sha256','wrong'),('expValues',1),('tensorCases',1),('refusals',0),('maximumDoubleError',float('nan'))]:
            with self.subTest(key=key):
                w=worker(0);w[key]=value
                with self.assertRaises(ValueError):validate_worker(w,'control',0)
        for collection in (workers()[:-1],workers()+[worker(0)],workers()[::-1]):
            with self.assertRaises(ValueError):evaluate(collection,'control',TELEMETRY)
    def test_geometry_alias_and_data_refusals(self):
        for key,value in [('rows',1),('columns',30),('iterations',128),('conditioning_calls',0),('conditioning_calls',32769),('implementation','ZeroBlocks.Kernels.Adaptive'),('mask_sha256','wrong'),('output_sha256','wrong'),('samples',[])]:
            with self.subTest(key=key):
                w=worker(0);row(w,'8','duplicate')[key]=value
                with self.assertRaises(ValueError):validate_worker(w,'control',0)
        w=worker(0);w['records'].pop()
        with self.assertRaises(ValueError):validate_worker(w,'control',0)
        for key,value in [('ticks',0),('thread_ns',-1),('process_ns',True),('gc',[0,-1,0]),('gc',[0,0])]:
            w=worker(0);w['records'][0]['samples'][0][key]=value
            with self.assertRaises(ValueError):validate_worker(w,'control',0)
    def test_control_single_bad_visit_despite_good_aggregate(self):
        for mode in ('duplicate','probe'):
            ws=workers();change(ws[0],'8',mode,103_000_000);v=evaluate(ws,'control',TELEMETRY)
            self.assertFalse(v['passed']);self.assertTrue(v['results']['8']['control_criteria'][mode]['aggregate_within_1_percent'])
    def test_control_bad_aggregate_despite_each_visit_within_bound(self):
        ws=workers()
        for w in ws:change(w,'30','probe',101_500_000)
        v=evaluate(ws,'control',TELEMETRY);self.assertFalse(v['passed']);self.assertTrue(v['results']['30']['control_criteria']['probe']['every_worker_within_2_percent'])
    def test_comparison_keeps_duplicate_gate(self):
        ws=workers('compare');change(ws[0],'8','duplicate',103_000_000);v=evaluate(ws,'compare',TELEMETRY)
        self.assertFalse(v['passed']);self.assertTrue(v['candidate_passed'])
    def test_short_batch_and_gc_fail_without_exclusion(self):
        for key,value in [('ticks',19_999_999),('gc',[1,0,0])]:
            ws=workers();row(ws[0],'8','actual')['samples'][0][key]=value
            v=evaluate(ws,'control',TELEMETRY);self.assertFalse(v['passed']);self.assertEqual(len(ws[0]['records'][0]['samples']),9)
    def test_foreign_cpu_and_steal_fail(self):
        for key,value in [('maximum_foreign_cpu_fraction',.0201),('maximum_steal_fraction',.0051)]:
            telemetry=TELEMETRY|{key:value};self.assertFalse(evaluate(workers(),'control',telemetry)['passed'])
    def test_candidate_single_unmasked_visit_failure(self):
        ws=workers('compare');change(ws[0],'8','probe',106_000_000);v=evaluate(ws,'compare',TELEMETRY)
        self.assertFalse(v['passed']);self.assertTrue(v['results']['8']['candidate_criteria']['aggregate_regression_at_most_2_percent'])
    def test_candidate_padded_single_visit_failure(self):
        ws=workers('compare');change(ws[0],'pad128','probe',86_000_000);v=evaluate(ws,'compare',TELEMETRY)
        self.assertFalse(v['passed']);self.assertTrue(v['results']['pad128']['candidate_criteria']['aggregate_actual_gain_at_least_15_percent'])
    def test_actual_copy_disagreement_failure(self):
        ws=workers('compare')
        for w in ws:change(w,'pad128','actual',96_000_000)
        self.assertFalse(evaluate(ws,'compare',TELEMETRY)['passed'])

if __name__=='__main__':unittest.main()
