import copy,unittest
import numpy as np
from common import BANKS,CORE,VARIANTS,order
from audit import inspect_timing,summarize,resources
from data import scalar

class AuditTests(unittest.TestCase):
    def fixture(self,visit=0):
        descriptions=[];cases=[]
        for d in BANKS:
            description={k:d[k] for k in ['name','repeats','diagnostic','width','rows']}
            description.update(input_sha256='1'*64,scale_sha256='2'*64,bias_sha256='3'*64,output_sha256=None if d['diagnostic'] else '4'*64)
            descriptions.append(description);case=dict(description,nodes=25,values=25*d['width']*d['rows'],held_unchanged=True,output_sha256='4'*64)
            for label,cycles in [('first',1),('warmup',16),('measured',48)]:
                records=[]
                for cycle in range(cycles):
                    for position in range(4):
                        variant=position if label=='first' else (visit+cycle+position)%4;repeats=1 if label=='first' else d['repeats']
                        records.append(dict(cycle=-1 if label=='first' else cycle,position=position,variant=VARIANTS[variant],repeats=repeats,
                            ticks=(950000 if variant==3 else 1000000)*repeats,allocated=0,gc=[0,0,0],output_sha256='4'*64))
                case[label]=records
            cases.append(case)
        value=dict(schema=1,protocol='layernorm-complete-bank-v1',visit=visit,frequency=1000000000,case_order=order(visit),results=cases,
            identity=dict(mode='run',runtime='10.0.8',core_sha256=CORE,probe_sha256='probe',affinity=4,vector_width=8,settings={},avx512=True,vector512_hardware=True))
        return value,descriptions

    def test_complete_schedule_and_record_refusals(self):
        value,descriptions=self.fixture();self.assertEqual(len(inspect_timing(value,0,dict(sha256='probe'),descriptions)),36)
        for key,wrong in [('ticks',0),('ticks',1.5),('variant','Product'),('repeats',1),('position',7),('gc',[0,-1,0]),('output_sha256','0'*64)]:
            broken=copy.deepcopy(value);broken['results'][0]['measured'][3][key]=wrong
            with self.subTest(key=key),self.assertRaises(AssertionError):inspect_timing(broken,0,dict(sha256='probe'),descriptions)
        for key,wrong in [('nodes',24),('width',385),('held_unchanged',False),('values',0),('input_sha256','wrong')]:
            broken=copy.deepcopy(value);broken['results'][0][key]=wrong
            with self.subTest(key=key),self.assertRaises(AssertionError):inspect_timing(broken,0,dict(sha256='probe'),descriptions)
        broken=copy.deepcopy(value);broken['results'][0]['measured'].pop()
        with self.assertRaises(AssertionError):inspect_timing(broken,0,dict(sha256='probe'),descriptions)
        broken=copy.deepcopy(value);broken['case_order'].reverse()
        with self.assertRaises(AssertionError):inspect_timing(broken,0,dict(sha256='probe'),descriptions)
        broken=copy.deepcopy(value);broken['identity']['settings']={'DOTNET_TieredCompilation':'0'}
        with self.assertRaises(AssertionError):inspect_timing(broken,0,dict(sha256='probe'),descriptions)

    def test_fixed_gain_control_and_worker_screens(self):
        rows=[]
        for visit in range(4):
            value,descriptions=self.fixture(visit);rows+=inspect_timing(value,visit,dict(sha256='probe'),descriptions)
        self.assertTrue(summarize(rows)['overall_passed'])
        broken=copy.deepcopy(rows)
        for row in broken:
            if row['variant']=='CopyB':row['mean_ms']*=1.1
        self.assertFalse(summarize(broken)['controls_passed'])
        broken=copy.deepcopy(rows)
        for row in broken:
            if row['case']=='e5-128tok' and row['variant']=='Wide':row['mean_ms']=.99
        self.assertTrue(summarize(broken)['controls_passed']);self.assertFalse(summarize(broken)['candidate_screen_passed'])
        broken=copy.deepcopy(rows)
        for row in broken:
            if row['case']==BANKS[-1]['name'] and row['visit']==2 and row['variant']=='Wide':row['mean_ms']=1.03
        self.assertFalse(summarize(broken)['candidate_screen_passed'])
        with self.assertRaises(AssertionError):summarize(rows[:-1]+[rows[0]])

    def test_resource_guards_and_births(self):
        run=dict(code=0,terminal_members=True,seconds=1.,started=1.,ended=2.,samples=2,members={'123':1.},child=dict(pid=123,birth=1.),peak_rss=400)
        samples=[dict(seconds=t,available=4*1024**3,members=[dict(pid=123,birth=1.,rss=400,affinity=[2],cpu=t)]) for t in [.1,.6]]
        self.assertEqual(resources(run,samples)['peak_rss'],400)
        for key,wrong in [('birth',2.),('affinity',[0]),('rss',4*1024**3),('cpu',float('nan'))]:
            broken=copy.deepcopy(samples);broken[-1]['members'][0][key]=wrong
            with self.subTest(key=key),self.assertRaises(AssertionError):resources(run,broken)
        broken=copy.deepcopy(samples);broken[-1]['available']=1024
        with self.assertRaises(AssertionError):resources(run,broken)

    def test_independent_scalar_reference(self):
        x=np.array([[1.,3.]],dtype='<f4');scale=np.array([2.,4.],dtype='<f4');bias=np.array([.5,-.5],dtype='<f4')
        np.testing.assert_array_equal(scalar([(x,scale,bias,0.,None)]),[-1.5,3.5])
        np.testing.assert_array_equal(scalar([(x,scale,None,0.,None)]),[-2.,4.])

if __name__=='__main__':unittest.main()
