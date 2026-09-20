from pathlib import Path
import copy,hashlib,tempfile,unittest
import numpy as np
from common import schedule,header
from audit import array,instrumentation

class TraceTests(unittest.TestCase):
    def test_fixed_complete_schedule(self):
        spec=dict(requests=[dict(original_request=i,name=name) for i,name in [(0,'first'),(10,'second'),(9,'third'),(20,'first')]])
        jobs=schedule(spec);self.assertEqual(len(jobs),8);self.assertEqual(len({j['id'] for j in jobs}),8)
        for index,value in [(1,9),(3,0)]:
            bad=copy.deepcopy(spec);bad['requests'][index]['original_request']=value
            with self.assertRaises(AssertionError):schedule(bad)

    def test_incomplete_or_wrong_worker_refused(self):
        job=dict(engine='managed',request=0)
        row=dict(complete=True,engine='managed',request_index=0,manifest_sha256='manifest',flags={},records=[dict(kind=k,inputs_unchanged=True,held_outputs_unchanged=True,outputs=[{}]*41) for k in ['MM','MN']])
        header(row,job,'manifest')
        for key,value in [('complete',False),('engine','native'),('request_index',1),('manifest_sha256','other'),('flags',{'DOTNET_TieredCompilation':'0'})]:
            bad=copy.deepcopy(row);bad[key]=value
            with self.assertRaises(AssertionError):header(bad,job,'manifest')
        bad=copy.deepcopy(row);bad['records'][0]['outputs'].pop()
        with self.assertRaises(AssertionError):header(bad,job,'manifest')

    def test_full_array_integrity_and_shape(self):
        values=np.array([1.,2.,3.,4.],dtype='<f4');description=dict(index=0,name='output',shape=[1,2,2])
        record=dict(**description,file='output.f32',values=4,sha256=hashlib.sha256(values.tobytes()).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'output.f32';path.write_bytes(values.tobytes());array(path,record,description)
            for key,value in [('values',3),('name','wrong'),('shape',[4]),('sha256','changed')]:
                bad=copy.deepcopy(record);bad[key]=value
                with self.assertRaises(AssertionError):array(path,bad,description)
            path.write_bytes(np.array([1.,2.,3.,5.],dtype='<f4').tobytes())
            with self.assertRaises(AssertionError):array(path,record,description)

    def test_instrumentation_is_measured_not_waived(self):
        reference=np.array([1.,2.],dtype='<f4');actual=np.array([1.01,2.],dtype='<f4')
        raw=lambda value:hashlib.sha256(value.tobytes()).hexdigest()
        record=dict(baseline_sha256=raw(reference),final_sha256=raw(actual),bitwise=False,max_scaled=float(actual[0])-1,failed_values=1)
        self.assertEqual(instrumentation(actual,reference,record)['failed_values'],1)
        for key,value in [('bitwise',True),('failed_values',0),('max_scaled',0.)]:
            bad=copy.deepcopy(record);bad[key]=value
            with self.assertRaises(AssertionError):instrumentation(actual,reference,bad)

if __name__=='__main__':unittest.main()
