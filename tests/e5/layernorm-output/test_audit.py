from pathlib import Path
from unittest.mock import patch
import copy,unittest
import audit as a

BASE=Path(__file__).resolve().parents[3]/'artifacts/e5-layernorm-output-20260920'

class Refusals(unittest.TestCase):
    def test_synthetic_identity_and_coverage(self):
        value=a.read(BASE/'proof/proof.json');self.assertEqual(a.synthetic(value['records']),790)
        for mutation in [lambda v:v[0].update(input_sha256='0'*64),lambda v:v[0].update(block=2),lambda v:v[1].update(has_bias=False),
                         lambda v:v[600].update(scale_sha256='0'*64),lambda v:v[789].update(bias_sha256='0'*64)]:
            records=copy.deepcopy(value['records']);mutation(records)
            with self.assertRaises(AssertionError):a.synthetic(records)

    def test_actual_proof_and_capture_metadata_refusals(self):
        proof_path=BASE/'proof/proof.json';capture_path=BASE/'capture/capture.json';read=a.read
        alterations=[(proof_path,lambda v:v.update(passed=False)),(proof_path,lambda v:v.update(cases=914)),
                     (proof_path,lambda v:v.update(comparisons=0)),(proof_path,lambda v:v.update(captured=124)),
                     (proof_path,lambda v:v['identity'].update(avx512=True)),(proof_path,lambda v:v['records'][0].update(guards=False)),
                     (proof_path,lambda v:v['records'][0].update(inputs_preserved=False)),(proof_path,lambda v:v['records'][0].update(inplace=False)),
                     (proof_path,lambda v:v['records'][0].update(scalar_error=1)),(proof_path,lambda v:v['records'][790].update(product_sha256='0'*64)),
                     (capture_path,lambda v:v['identity'].update(core_sha256='0'*64)),(capture_path,lambda v:v['cases'].reverse())]
        for target,mutate in alterations:
            value=read(target);mutate(value)
            with patch.object(a,'read',side_effect=lambda p:value if p==target else read(p)):
                with self.assertRaises(AssertionError):a.audit(BASE)

    def test_full_array_geometry_and_nonfinite_refusals(self):
        for values in [a.np.array([1.],dtype='<f4'),a.np.array([a.np.nan,1.],dtype='<f4'),a.np.array([a.np.inf,1.],dtype='<f4')]:
            with patch.object(a.np,'fromfile',return_value=values):
                with self.assertRaises(AssertionError):a.array(BASE/'unused.f32',2)

if __name__=='__main__':unittest.main()
