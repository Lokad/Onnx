import unittest
from prepare import changed, PRIOR, TARGET, INSERTION, ANCHOR, TOOLS


class CandidateScope(unittest.TestCase):
    def test_insertion_reverses_exactly_and_preserves_segmented_priority(self):
        original = (PRIOR/'source'/TARGET).read_text(encoding='utf8')
        candidate = changed(original)
        self.assertEqual(candidate.replace(INSERTION, ''), original)
        self.assertLess(candidate.index('if (options.UseSegmentedConvolution'), candidate.index(INSERTION))
        self.assertLess(candidate.index(INSERTION), candidate.index(ANCHOR))

    def test_rejects_duplicate_or_changed_source(self):
        original = (PRIOR/'source'/TARGET).read_text(encoding='utf8')
        with self.assertRaises(AssertionError): changed(changed(original))
        with self.assertRaises(AssertionError): changed(original.replace(ANCHOR, ''))

    def test_no_shared_dispatch_pool_or_compiler_annotations(self):
        helper = (TOOLS/'DirectDepthwise.cs.txt').read_text(encoding='utf8')
        for forbidden in ['MethodImpl', 'ArrayPool', 'Parallel.', 'MatMul2D(', 'Im2col', 'new DenseTensor']:
            self.assertNotIn(forbidden, helper)


if __name__ == '__main__': unittest.main()
