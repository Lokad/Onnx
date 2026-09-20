from pathlib import Path
import copy,hashlib,tempfile,unittest
import numpy as np
from common import schedule,header
from audit import array,instrumentation,input_record,decompose,original

class CrossTests(unittest.TestCase):
    def spec(self):
        return dict(requests=[dict(selected_request=i,original_request=[0,10,9,20][i],features=f,name=['first','second','third','first'][i]) for i in range(4) for f in ['managed','native']])

    def test_complete_schedule_and_same_input_mapping(self):
        spec=self.spec();jobs=schedule(spec);self.assertEqual(len(jobs),16);self.assertEqual(len({j['id'] for j in jobs}),16)
        for key,value in [('original_request',9),('features','native'),('selected_request',1)]:
            bad=copy.deepcopy(spec);bad['requests'][0][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):schedule(bad)
        item=dict(managed_input={'hash':'managed'},native_input={'hash':'native'})
        for kind in ['MM','NM']:self.assertEqual(input_record(item,kind),item['managed_input'])
        for kind in ['MN','NN']:self.assertEqual(input_record(item,kind),item['native_input'])
        with self.assertRaises(AssertionError):input_record(item,'bad')

    def test_wrong_or_incomplete_worker_refused(self):
        job=dict(engine='native',request=0)
        row=dict(complete=True,engine='native',request_index=0,manifest_sha256='manifest',flags={},records=[dict(kind=k,inputs_unchanged=True,held_outputs_unchanged=True,outputs=[{}]*12) for k in ['NM','NN']])
        header(row,job,'manifest')
        for key,value in [('complete',False),('engine','managed'),('request_index',1),('manifest_sha256','changed'),('flags',{'LOKAD_ROOT':'wrong'})]:
            bad=copy.deepcopy(row);bad[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):header(bad,job,'manifest')
        for change in [lambda r:r['records'].reverse(),lambda r:r['records'][0]['outputs'].pop(),lambda r:r['records'][1].update(held_outputs_unchanged=False)]:
            bad=copy.deepcopy(row);change(bad)
            with self.assertRaises(AssertionError):header(bad,job,'manifest')

    def test_array_integrity_and_shape(self):
        values=np.array([1.,2.,3.,4.],dtype='<f4');description=dict(index=0,name='output',shape=[1,2,2])
        record=dict(**description,file='output.f32',values=4,sha256=hashlib.sha256(values.tobytes()).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'output.f32';path.write_bytes(values.tobytes());array(path,record,description)
            for key,value in [('values',3),('name','wrong'),('shape',[4]),('sha256','changed')]:
                bad=copy.deepcopy(record);bad[key]=value
                with self.assertRaises(AssertionError):array(path,bad,description)
            path.write_bytes(np.array([1.,2.,3.,5.],dtype='<f4').tobytes())
            with self.assertRaises(AssertionError):array(path,record,description)

    def test_extraction_failure_cannot_be_called_equality(self):
        reference=np.array([1.,2.],dtype='<f4');actual=np.array([1.01,2.],dtype='<f4');raw=lambda v:hashlib.sha256(v.tobytes()).hexdigest()
        record=dict(baseline_sha256=raw(reference),final_sha256=raw(actual),bitwise=False,max_scaled=float(actual[0])-1,failed_values=1)
        self.assertEqual(instrumentation(actual,reference,record)['failed_values'],1)
        for key,value in [('bitwise',True),('failed_values',0),('max_scaled',0.)]:
            bad=copy.deepcopy(record);bad[key]=value
            with self.assertRaises(AssertionError):instrumentation(actual,reference,bad)

    def test_common_denominator_and_independent_decomposition(self):
        cells={k:np.array([[v]],dtype='<f4') for k,v in [('MM',2),('MN',101),('NM',1),('NN',100)]}
        result=decompose(cells)
        self.assertEqual(result['closure_max'],[0.,0.]);self.assertEqual(result['terms']['diagonal_MM-NN']['max_scaled'],.98)
        self.assertEqual(result['terms']['engine_MM-NM']['max_scaled'],.01)
        self.assertEqual(result['pairwise_native_reference']['managed_incoming']['max_scaled'],1.)
        self.assertEqual(result['terms']['interaction']['max_scaled'],0.)
        self.assertEqual(result['terms']['diagonal_MM-NN']['maximum_index'],[0,0])
        for bad in [dict(cells,NN=np.array([100.],dtype='<f4')),dict(cells,MM=np.array([[np.nan]],dtype='<f4'))]:
            with self.assertRaises(AssertionError):decompose(bad)

    def test_cut_resource_bounds_and_process_identity(self):
        limits=dict(seconds=180,rss=8*1024**3,available=1024**3,preflight_available=10*1024**3)
        state=dict(complete=True,code=0,terminal_members=True,limits=limits,preflight_available=11*1024**3,seconds=1.,started=1.,ended=2.,
            samples=2,peak_rss=400,members={'123':1.},child=dict(pid=123,birth=1.),supervisor=dict(pid=100,birth=.5))
        samples=[dict(seconds=t,available=2*1024**3,members=[dict(pid=123,birth=1.,rss=400,affinity=[2])]) for t in [.1,.6]]
        self.assertEqual(original.resources(state,samples,dict(limits=limits))['peak_rss'],400)
        for key,value in [('seconds',181),('preflight_available',9*1024**3),('terminal_members',False)]:
            bad=copy.deepcopy(state);bad[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):original.resources(bad,samples,dict(limits=limits))
        for key,value in [('birth',2.),('affinity',[0]),('rss',9*1024**3)]:
            bad=copy.deepcopy(samples);bad[-1]['members'][0][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):original.resources(state,bad,dict(limits=limits))
        bad=copy.deepcopy(samples);bad[-1]['available']=1024
        with self.assertRaises(AssertionError):original.resources(state,bad,dict(limits=limits))

if __name__=='__main__':unittest.main()
