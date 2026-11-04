"""Independent complete-array checks with small deterministic binary fixtures."""
import hashlib,json,struct
from pathlib import Path
import tempfile
import unittest

import graph_worker as graph


class CompleteArrayTests(unittest.TestCase):
    def result(self,root,values,shape=None,stochastic=False):
        root.mkdir()
        data=struct.pack('<'+'f'*len(values),*values)
        (root/'0.f32').write_bytes(data)
        (root/'result.json').write_text(json.dumps(dict(stochastic_outputs=stochastic,arrays=[dict(name='output',shape=shape or [1,len(values)],
            values=len(values),file='0.f32',sha256=hashlib.sha256(data).hexdigest())])))

    def test_checks_every_float_with_native_scaled_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.result(root/'managed',[.5,-1,2.0001]);self.result(root/'native',[.5,-1,2])
            value=graph.compare(root/'managed',root/'native')
            self.assertEqual(3,value['values'])
            self.assertGreater(value['maximum_scaled_error'],0)
            self.assertLess(value['maximum_scaled_error'],1e-4)

    def test_nonfinite_wrong_shape_and_out_of_bound_values_fail(self):
        for values,shape in [([.5,-1,float('nan')],None),([.5,-1,2],[3,1]),([.5,-1,2.01],None)]:
            with self.subTest(values=values,shape=shape),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                self.result(root/'managed',values,shape);self.result(root/'native',[.5,-1,2])
                with self.assertRaises(ValueError):graph.compare(root/'managed',root/'native')

    def test_modified_full_array_cannot_reuse_old_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.result(root/'managed',[.5,-1,2]);self.result(root/'native',[.5,-1,2])
            (root/'managed/0.f32').write_bytes(struct.pack('<fff',.5,-1,2.00001))
            with self.assertRaisesRegex(ValueError,'array changed'):graph.compare(root/'managed',root/'native')

    def test_explicit_stochastic_outputs_check_shape_finiteness_and_hashes_without_claiming_equality(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.result(root/'managed',[.1,.2,.3],stochastic=True)
            self.result(root/'native',[.9,.8,.7],stochastic=True)
            value=graph.compare(root/'managed',root/'native')
            self.assertEqual(3,value['values'])
            self.assertIsNone(value['maximum_scaled_error'])
            self.assertTrue(value['stochastic_outputs'])
            (root/'managed/0.f32').write_bytes(struct.pack('<fff',.1,.2,.31))
            with self.assertRaisesRegex(ValueError,'array changed'):graph.compare(root/'managed',root/'native')

    def test_stochastic_mode_must_match_and_never_accepts_nonfinite_values(self):
        for flag,values in [(False,[.9,.8,.7]),(True,[.9,float('nan'),.7])]:
            with self.subTest(flag=flag),tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                self.result(root/'managed',[.1,.2,.3],stochastic=True)
                self.result(root/'native',values,stochastic=flag)
                with self.assertRaises(ValueError):graph.compare(root/'managed',root/'native')

    def test_custom_schedule_requires_declared_measurements_and_warmup_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.result(root/'worker',[.1,.2,.3],stochastic=True)
            path=root/'worker/result.json'
            value=json.loads(path.read_text())
            value.update(key='custom',mode='timing',passed=True,inputs_unchanged=True,held_outputs_unchanged=True,
                         native_versions={},warmup_elapsed_seconds=10.,calls=7,
                         clocks=[dict(index=i,warmup=i<4,ticks=10,frequency=1000) for i in range(7)])
            value['arrays'][0]['max_scaled_error']=None
            case=dict(key='custom',stochastic_outputs=True,outputs=[dict(name='output',shape=[1,3])],
                      timing=dict(warmup_calls=4,warmup_seconds=10,measured_calls=3))
            manifest=dict(cases=[case])
            path.write_text(json.dumps(value))
            _,samples=graph.validate(root/'worker',{},'custom','ort','timing',manifest,{})
            self.assertEqual([10.,10.,10.],samples)
            for bad in ('short-warmup','missing-measurement','interleaved-warmup','false-equality'):
                broken=json.loads(json.dumps(value))
                if bad=='short-warmup':broken['warmup_elapsed_seconds']=9.9
                elif bad=='missing-measurement':broken['clocks'].pop();broken['calls']-=1
                elif bad=='interleaved-warmup':broken['clocks'][1]['warmup']=False;broken['clocks'][5]['warmup']=True
                else:broken['arrays'][0]['max_scaled_error']=0.
                path.write_text(json.dumps(broken))
                with self.subTest(bad=bad),self.assertRaises(ValueError):
                    graph.validate(root/'worker',{},'custom','ort','timing',manifest,{})


if __name__=='__main__':unittest.main()
