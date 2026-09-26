"""Only the admitted exact product pair may enter fresh graph comparison."""
import unittest
from protocol import read
from prepare import PRIOR,validate


class Prerequisites(unittest.TestCase):
    def fixture(self):return {label:read(folder/'analysis.json') for label,folder in PRIOR.items()}

    def test_actual_closed_inputs(self):validate(self.fixture())

    def test_different_current_root_rejected(self):
        value=self.fixture();value['models']['identities']['selected']['Lokad.Onnx.dll']['sha256']='0'*64
        with self.assertRaises(AssertionError):validate(value)

    def test_failed_application_rejected(self):
        value=self.fixture();value['app']['performance']['admitted']=False
        with self.assertRaises(AssertionError):validate(value)

    def test_missing_application_control_rejected(self):
        value=self.fixture();value['app']['performance']['controls'].pop()
        with self.assertRaises(AssertionError):validate(value)

    def test_changed_pyannote_public_result_rejected(self):
        value=self.fixture();value['pyannote']['results']['candidate']['complete_public_results_exact']=False
        with self.assertRaises(AssertionError):validate(value)


if __name__=='__main__':unittest.main()
