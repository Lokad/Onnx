"""Reject consumers or product transitions outside the closed compatibility proof."""
import unittest
from consumer_reuse import PREVIOUS, MODELS, read, validate


class ConsumerReuse(unittest.TestCase):
    def fixture(self):
        return [read(PREVIOUS/'analysis.json'), read(MODELS/'analysis.json'),
                read(MODELS/'collected/evidence/compatibility.json'), read(PREVIOUS/'collected/built.json')]

    def test_actual_closed_consumer(self):
        result = validate(*self.fixture())
        self.assertFalse(result['consumer_rebuilt'])
        self.assertTrue(result['fresh_complete_model_checks_required'])

    def test_different_consumer_is_rejected(self):
        args = self.fixture(); args[-1]['consumer']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): validate(*args)

    def test_wrong_previously_qualified_product_is_rejected(self):
        args = self.fixture(); args[2]['qualified_model_product']['Lokad.Onnx.dll']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): validate(*args)

    def test_wrong_new_data_product_is_rejected(self):
        args = self.fixture(); args[1]['identities']['candidate']['Lokad.Onnx.Data.dll']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): validate(*args)

    def test_changed_data_behavior_is_rejected(self):
        args = self.fixture(); args[2]['all_data_methods_exact'] = False
        with self.assertRaises(AssertionError): validate(*args)

    def test_lost_flags_are_rejected(self):
        args = self.fixture(); args[2]['all_original_method_flags_preserved'] = False
        with self.assertRaises(AssertionError): validate(*args)

    def test_wrong_census_is_rejected(self):
        args = self.fixture(); args[2]['actual_model_census']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): validate(*args)


if __name__ == '__main__': unittest.main()
