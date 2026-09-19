import copy,hashlib,struct,unittest
from audit import SHAPES,MODES,WIDTHS,ITERATIONS,validate_worker,screen

def worker(order):
    rows=[]
    for i,name in enumerate(SHAPES):
        columns=WIDTHS[i];active=30 if name.startswith('pad') else columns
        mask=struct.pack('<f',0)*active+struct.pack('<I',0xff7fffff)*(columns-active)
        for mode in MODES:
            cost=8 if name.startswith('pad') and mode in ('skip','adaptive') else 10
            rows.append(dict(name=name,mode=mode,rows=12*columns,columns=columns,iterations=ITERATIONS[i],conditioning_calls=100,
                input_sha256='input-'+name,mask_sha256=hashlib.sha256(mask).hexdigest(),output_sha256='output-'+name,
                samples=[dict(ticks=cost*ITERATIONS[i]*1_000_000,thread_ns=10,process_ns=10,gc=[0,0,0]) for _ in range(9)]))
    return dict(schema=1,order=order,checkOnly=False,width=8,expValues=2503712,tensorCases=918,refusals=4,maximumDoubleError=1e-7,
        core_sha256='7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9',frequency=1_000_000_000,runtime='.NET 10.0.8',records=rows)

class Audit(unittest.TestCase):
    def test_valid(self):validate_worker(worker(0),0)
    def test_coverage_and_identity_refusals(self):
        for key,value in [('order',1),('width',16),('tensorCases',917),('expValues',1),('refusals',0),('core_sha256','wrong'),('maximumDoubleError',1e-4),('frequency',0),('runtime','wrong')]:
            with self.subTest(key=key):
                w=worker(0);w[key]=value
                with self.assertRaises(ValueError):validate_worker(w,0)
    def test_row_and_sample_refusals(self):
        for key,value in [('iterations',1),('mask_sha256','wrong'),('output_sha256','wrong'),('conditioning_calls',0),('samples',[])]:
            with self.subTest(key=key):
                w=worker(0);w['records'][0][key]=value
                with self.assertRaises(ValueError):validate_worker(w,0)
        w=worker(0);w['records'][0]['samples'][0]['thread_ns']=-1
        with self.assertRaises(ValueError):validate_worker(w,0)
        w=worker(0);w['records'].reverse()
        with self.assertRaises(ValueError):validate_worker(w,0)
    def test_screen_passes_prospective_example(self):self.assertTrue(screen([worker(i) for i in range(8)])[0])
    def test_single_padded_visit_can_fail_despite_aggregate_gain(self):
        workers=[worker(i) for i in range(8)]
        for r in workers[0]['records']:
            if r['name']=='pad128' and r['mode']=='adaptive':
                for s in r['samples']:s['ticks']=int(8.6*r['iterations']*1e6)
        passed,table=screen(workers);self.assertFalse(passed);self.assertTrue(table['pad128']['criteria']['aggregate_actual_gain_at_least_15_percent'])
    def test_single_unmasked_regression_fails(self):
        workers=[worker(i) for i in range(8)]
        for r in workers[0]['records']:
            if r['name']=='128' and r['mode']=='adaptive':
                for s in r['samples']:s['ticks']=int(10.6*r['iterations']*1e6)
        passed,table=screen(workers);self.assertFalse(passed);self.assertTrue(table['128']['criteria']['aggregate_regression_at_most_2_percent'])
    def test_copy_control_disagreement_fails(self):
        workers=[worker(i) for i in range(8)]
        for w in workers:
            for r in w['records']:
                if r['name']=='pad128' and r['mode']=='copied':
                    for s in r['samples']:s['ticks']=int(10.4*r['iterations']*1e6)
        passed,table=screen(workers);self.assertFalse(passed);self.assertFalse(table['pad128']['criteria']['copy_actual_control_within_3_percent'])

if __name__=='__main__':unittest.main()
