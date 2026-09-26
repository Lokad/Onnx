import unittest
from scope import OLD, EDITS, instrument, references
from codegen import inspect


class ScopeTests(unittest.TestCase):
    def test_original_calls_and_validation_recover_exactly(self):
        before = (OLD / 'Screen.cs').read_text(encoding='utf8')
        after = instrument(before)
        for old, new in reversed(EDITS):
            after = after.replace(new, old)
        self.assertEqual(after, before)

    def test_missing_clock_anchor_rejected(self):
        before = (OLD / 'Screen.cs').read_text(encoding='utf8')
        with self.assertRaises(AssertionError):
            instrument(before.replace('long ticks=Stopwatch.GetTimestamp()-start;', 'long ticks=1;'))

    def test_closed_product_and_relevant_compiled_scope(self):
        products, evidence = references()
        self.assertTrue(products['current']['sha256'].startswith('8bb22038'))
        self.assertTrue(products['candidate']['sha256'].startswith('946ddfb6'))
        self.assertGreater(evidence['compiled_scope']['unchanged_core_methods'], 100)

    def test_truncated_listing_rejected(self):
        with self.assertRaises(AssertionError):
            inspect('; Assembly listing for method X (Tier1)\n')


if __name__ == '__main__':
    unittest.main()
