"""Reject incomplete or retrospectively extended elapsed-time priming evidence."""
import copy
import unittest
from prefix import validate
from prepare import ROOT, consumer
from protocol import read


class PrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = ROOT / 'artifacts/parakeet-pad-warmup-diagnostic-amd-20260926/collected/current-capture'
        cls.prefix = read(base / 'priming.json')
        cls.suffix = read(base / 'result.json')
        cls.suffix['suffixStart'] = cls.suffix['rows'][0]['clocks'][0]['start'] - 1

    def test_retained_complete_prefix(self):
        self.assertEqual(validate(self.prefix, self.suffix)['calls'], 18720)

    def test_missing_call(self):
        prefix = copy.deepcopy(self.prefix)
        prefix['passes'][0]['rows'][0]['clocks'].pop()
        with self.assertRaises(AssertionError): validate(prefix, self.suffix)

    def test_changed_output_hash(self):
        prefix = copy.deepcopy(self.prefix)
        prefix['passes'][0]['rows'][0]['output'] = '0' * 64
        with self.assertRaises(AssertionError): validate(prefix, self.suffix)

    def test_short_interval(self):
        prefix = dict(self.prefix, frequency=self.prefix['frequency'] * 4)
        suffix = dict(self.suffix, frequency=prefix['frequency'])
        with self.assertRaises(AssertionError): validate(prefix, suffix)

    def test_extra_round_after_stopping_condition(self):
        prefix = copy.deepcopy(self.prefix)
        prefix['passes'].append(copy.deepcopy(prefix['passes'][-1]))
        prefix['calls'] += 9360
        with self.assertRaises(AssertionError): validate(prefix, self.suffix)

    def test_suffix_overlap(self):
        suffix = dict(self.suffix, suffixStart=self.prefix['ended'])
        with self.assertRaises(AssertionError): validate(self.prefix, suffix)

    def test_generated_suffix_and_prime_preserve_source(self):
        # consumer() checks the exact reversible edits against the retained source.
        text = consumer()
        self.assertEqual(text.count('static long Prime('), 1)
        self.assertEqual(text.count('var returned = CPUExecutionProvider.Pad('), 2)


if __name__ == '__main__': unittest.main()
