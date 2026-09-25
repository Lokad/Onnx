"""Keep failed release evidence visible without blocking exact-candidate diagnosis."""
import copy
import unittest
from prepare import ROOT,SOURCE,QUALIFICATION,diagnostic_context,read


class DiagnosticContext(unittest.TestCase):
    def setUp(self):
        self.product=read(SOURCE/'prepared.json')['product']
        self.proofs={k:read(v/'closed.json') for k,v in QUALIFICATION.items() if k!='app'}
        self.results={k:read(v/'analysis.json') for k,v in QUALIFICATION.items() if k!='app'}
        # The new direct comparison is still live. This fixture combines the
        # actual released identity and exact M78 identity, not timing evidence.
        old=read(ROOT/'artifacts/parakeet-slice-dense-conversion-app-amd-20260925/analysis.json')
        self.results['app']=dict(passed=True,identities=dict(
            current=old['identities']['current'],candidate=copy.deepcopy(self.product)))
        self.proofs['app']=dict(passed=True,admitted=False)
        self.graph_products=read(QUALIFICATION['graphs']/'payload.json')['products']

    def check(self):
        return diagnostic_context(self.product,self.proofs,self.results,self.graph_products)

    def test_actual_graph_failure_stays_failed_in_diagnosis(self):
        value=self.check()
        self.assertEqual(value['failed_graph_cases'],['e5-8tok'])
        self.assertTrue(value['graph_controls_passed'])
        self.assertTrue(value['isolated_candidate'])
        for key in ['release_admitted','root_product_changed','graph_admitted','application_admitted']:
            self.assertFalse(value[key],key)

    def test_failed_numerical_audit_is_rejected(self):
        self.results['app']['passed']=False
        with self.assertRaises(AssertionError):self.check()

    def test_wrong_candidate_core_is_rejected(self):
        self.results['app']['identities']['candidate']['Lokad.Onnx.dll']['sha256']='wrong'
        with self.assertRaises(AssertionError):self.check()

    def test_old_data_without_owned_preparation_is_rejected(self):
        self.results['app']['identities']['candidate']['Lokad.Onnx.Data.dll']=self.results['app']['identities']['current']['Lokad.Onnx.Data.dll']
        with self.assertRaises(AssertionError):self.check()

    def test_intermediate_baseline_cannot_be_called_release(self):
        self.results['app']['identities']['current']=self.results['candidate_app']['identities']['current']
        with self.assertRaises(AssertionError):self.check()

    def test_favorable_graph_relabel_is_rejected(self):
        self.proofs['graphs']['admitted']=True
        with self.assertRaises(AssertionError):self.check()

    def test_graph_from_another_product_is_rejected(self):
        self.graph_products['candidate']['Lokad.Onnx.dll']['sha256']='wrong'
        with self.assertRaises(AssertionError):self.check()

    def test_original_candidate_must_remain_admitted(self):
        self.proofs['candidate_app']['admitted']=False
        with self.assertRaises(AssertionError):self.check()


if __name__=='__main__':unittest.main()
