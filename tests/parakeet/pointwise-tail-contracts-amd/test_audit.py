"""Ensure numerical failures cannot disappear behind process completion."""
import copy
import unittest
from audit import validate_cases


class ContractAuditTests(unittest.TestCase):
    def setUp(self):
        self.case=dict(m=66,n=64,k=61,exceptional=True,oracle=True)
        self.spec=dict(raw_cases=[self.case])
        self.result=dict(completed=True,passed=True,failed=0,results=[dict(self.case,
            bit_exact=True,inputs_immutable=True,guards_intact=True,allocated_bytes=0,output_sha256='a'*64)])

    def test_records_failure_without_passing_candidate(self):
        self.assertEqual(validate_cases(self.result,self.spec,True),[])
        failed=dict(self.case,bit_exact=False,error='mismatched NaN payload')
        self.result.update(passed=False,failed=1,results=[failed])
        self.assertEqual(validate_cases(self.result,self.spec,True),[failed])
        self.result['passed']=True
        with self.assertRaises(AssertionError):validate_cases(self.result,self.spec,True)

    def test_rejects_missing_wrong_or_mutating_cases(self):
        for change in [lambda r:r['results'].clear(),
            lambda r:r['results'][0].update(k=63),
            lambda r:r['results'][0].update(allocated_bytes=16),
            lambda r:r['results'][0].update(guards_intact=False),
            lambda r:r['results'][0].update(inputs_immutable=False)]:
            with self.subTest(change=change):
                result=copy.deepcopy(self.result);change(result)
                with self.assertRaises(AssertionError):validate_cases(result,self.spec,True)


if __name__=='__main__':unittest.main()
