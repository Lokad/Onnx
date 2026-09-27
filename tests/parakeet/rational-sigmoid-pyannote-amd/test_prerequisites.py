"""Reject mismatched products or incomplete application admission."""
import unittest
from prepare import PRIOR,validate
from protocol import read


class Prerequisites(unittest.TestCase):
    def fixture(self):
        return {name:read(folder/'analysis.json') for name,folder in PRIOR.items()}

    def test_closed_prerequisites(self):
        validate(self.fixture())

    def test_wrong_product(self):
        value=self.fixture();value['product']['identities']['candidate']['Lokad.Onnx.dll']['sha256']='0'*64
        with self.assertRaises(AssertionError):validate(value)

    def test_wrong_shared_data(self):
        value=self.fixture();value['shared']['identities']['candidate']['Lokad.Onnx.Data.dll']['sha256']='0'*64
        with self.assertRaises(AssertionError):validate(value)

    def test_missing_control(self):
        value=self.fixture();value['app']['performance']['controls'].pop()
        with self.assertRaises(AssertionError):validate(value)

    def test_failed_gate(self):
        value=self.fixture();value['app']['performance']['gates'][0]['passed']=False
        with self.assertRaises(AssertionError):validate(value)

    def test_no_application_admission(self):
        value=self.fixture();value['app']['performance']['admitted']=False
        with self.assertRaises(AssertionError):validate(value)


if __name__=='__main__':unittest.main()
