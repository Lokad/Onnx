"""Independent complete-array checks with small deterministic binary fixtures."""
import hashlib,json,struct
from pathlib import Path
import tempfile
import unittest

import graph_worker as graph


class CompleteArrayTests(unittest.TestCase):
    def result(self,root,values,shape=None):
        root.mkdir()
        data=struct.pack('<'+'f'*len(values),*values)
        (root/'0.f32').write_bytes(data)
        (root/'result.json').write_text(json.dumps(dict(arrays=[dict(name='output',shape=shape or [1,len(values)],
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


if __name__=='__main__':unittest.main()
