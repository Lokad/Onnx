import copy
import unittest
from audit import validate_raw


class ArithmeticAuditTests(unittest.TestCase):
    def setUp(self):
        case=dict(m=64,n=64,k=33,exceptional=True,oracle=True)
        self.spec=dict(raw_cases=[case])
        self.result=dict(completed=True,comparison_selftests=9,passed=True,failed=0,results=[dict(case,
            arithmetic_contract_passed=True,non_nan_bits_exact=True,nan_classification_exact=True,nan_payload_differences=3,
            bit_exact=False,inputs_immutable=True,guards_intact=True,allocated_bytes=0,output_sha256='a'*64,baseline_output_sha256='b'*64)])

    def test_payload_changes_are_recorded_without_claiming_bit_identity(self):
        failures,payloads=validate_raw(self.result,self.spec)
        self.assertFalse(failures);self.assertEqual(payloads[0]['differences'],3)
        wrong=copy.deepcopy(self.result);wrong['results'][0]['bit_exact']=True
        with self.assertRaises(AssertionError):validate_raw(wrong,self.spec)

    def test_rejects_numeric_classification_ownership_or_coverage_changes(self):
        for change in [lambda r:r['results'][0].update(non_nan_bits_exact=False),lambda r:r['results'][0].update(nan_classification_exact=False),
            lambda r:r['results'][0].update(inputs_immutable=False),lambda r:r['results'][0].update(guards_intact=False),
            lambda r:r['results'][0].update(allocated_bytes=1),lambda r:r['results'].clear()]:
            with self.subTest(change=change):
                wrong=copy.deepcopy(self.result);change(wrong)
                with self.assertRaises(AssertionError):validate_raw(wrong,self.spec)

    def test_numerical_failure_cannot_pass(self):
        row=dict(self.spec['raw_cases'][0],arithmetic_contract_passed=False,bit_exact=False,error='Non-NaN bit mismatch')
        wrong=dict(self.result,results=[row],passed=False,failed=1)
        self.assertEqual(validate_raw(wrong,self.spec)[0],[row])
        wrong['passed']=True
        with self.assertRaises(AssertionError):validate_raw(wrong,self.spec)


if __name__=='__main__':unittest.main()
