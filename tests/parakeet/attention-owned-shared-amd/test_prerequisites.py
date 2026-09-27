"""Check eligibility failures using actual compiled evidence and the admitted application."""
import copy
import unittest
from protocol import pin, read
from prepare import MODELS, PREVIOUS, OLD, APP
from prerequisites import validate


class Prerequisites(unittest.TestCase):
    def fixture(self):
        compatible = read(MODELS/'collected/evidence/compatibility.json')
        models = read(MODELS/'analysis.json')
        app = read(APP/'analysis.json')
        return copy.deepcopy([compatible, models, read(PREVIOUS/'analysis.json'), app,
                              pin(OLD/'runtimes/baseline/Replay.dll')])

    def test_admission_for_actual_pair(self): validate(*self.fixture())

    def test_different_consumer(self):
        args = self.fixture(); args[-1]['sha256'] = '0'*64
        with self.assertRaises(AssertionError): validate(*args)

    def test_failed_application(self):
        args = self.fixture(); args[3]['performance']['admitted'] = False
        with self.assertRaises(AssertionError): validate(*args)

    def test_failed_control(self):
        args = self.fixture(); args[3]['performance']['controls'][0]['passed'] = False
        with self.assertRaises(AssertionError): validate(*args)

    def test_wrong_current(self):
        args = self.fixture(); args[3]['identities']['current']['Lokad.Onnx.dll']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): validate(*args)

    def test_wrong_model_census(self):
        args = self.fixture(); args[0]['actual_model_census']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): validate(*args)

    def test_unrelated_changed_method(self):
        args = self.fixture(); args[0]['compiled_scope'][0]['changed'][0] = 'Unrelated::Op::Void Op()'
        with self.assertRaises(AssertionError): validate(*args)


if __name__ == '__main__': unittest.main()
