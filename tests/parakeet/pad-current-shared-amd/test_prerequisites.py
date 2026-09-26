"""Reject incompatible reused consumers and failed application prerequisites."""
import copy
import unittest
from protocol import pin,read
from prepare import MODELS,PREVIOUS,OLD
from prerequisites import validate


class Prerequisites(unittest.TestCase):
    def fixture(self):
        compatible=read(MODELS/'collected/evidence/compatibility.json')
        models=read(MODELS/'analysis.json')
        previous=read(PREVIOUS/'analysis.json')
        # Exercise the binding seam before the independent application finishes.
        app=dict(passed=True,performance=dict(admitted=True),identities=dict(
            current=models['identities']['selected'],candidate=models['identities']['candidate']))
        return [compatible,models,previous,app,pin(OLD/'runtimes/baseline/Replay.dll')]

    def test_actual_consumer_has_reviewed_compatibility_chain(self):
        validate(*self.fixture())

    def test_different_consumer_is_rejected(self):
        args=self.fixture();args[-1]['sha256']='0'*64
        with self.assertRaises(AssertionError):validate(*args)

    def test_failed_application_is_rejected(self):
        args=self.fixture();args[3]['performance']['admitted']=False
        with self.assertRaises(AssertionError):validate(*args)

    def test_wrong_current_product_is_rejected(self):
        args=self.fixture();args[3]=copy.deepcopy(args[3])
        args[3]['identities']['current']['Lokad.Onnx.dll']['sha256']='0'*64
        with self.assertRaises(AssertionError):validate(*args)

    def test_removed_old_binding_is_rejected(self):
        args=self.fixture();args[0]['original_public_bindings_preserved']=False
        with self.assertRaises(AssertionError):validate(*args)

    def test_component_failure_cannot_disappear(self):
        args=self.fixture();args[0]['failed_component_controls'].pop()
        with self.assertRaises(AssertionError):validate(*args)


if __name__=='__main__':unittest.main()
