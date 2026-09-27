"""Check eligibility failures using actual compiled evidence and a hypothetical passed application."""
import copy
import unittest
from protocol import pin, read
from prepare import MODELS, PREVIOUS, OLD
from prerequisites import validate


class Prerequisites(unittest.TestCase):
    def fixture(self):
        compatible = read(MODELS/'collected/evidence/compatibility.json')
        models = read(MODELS/'analysis.json')
        # This synthetic admission is only a unit-test fixture, never evidence.
        app = dict(passed=True, identities=dict(current=models['identities']['selected'], candidate=models['identities']['candidate']),
                   performance=dict(admitted=True, controls=[dict(passed=True) for _ in range(63)],
                       gates=[dict(passed=True) for _ in range(20)]+[dict(passed=True, name='corpus-at-least-one-percent-gain', limit=.99)]))
        return copy.deepcopy([compatible, models, read(PREVIOUS/'analysis.json'), app,
                              pin(OLD/'runtimes/baseline/Replay.dll')])

    def test_hypothetical_admission_for_actual_pair(self): validate(*self.fixture())

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

    def test_lost_component_failure(self):
        args = self.fixture(); args[0]['failed_component_controls'].pop()
        with self.assertRaises(AssertionError): validate(*args)

    def test_unrelated_changed_method(self):
        args = self.fixture(); args[0]['compiled']['methods'][0]['changed'][0] = 'Unrelated::Op::Void Op()'
        with self.assertRaises(AssertionError): validate(*args)


if __name__ == '__main__': unittest.main()
