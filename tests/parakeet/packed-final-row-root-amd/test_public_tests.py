"""Reject assertion drift and misplaced hardware guards before public integration."""
import unittest
from public_tests import SOURCE,NAME,guarded,verify,qualified_census,FMA


class PublicTestScope(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.original=(SOURCE/'source'/NAME).read_text(encoding='utf8')

    def test_guards_preserve_every_existing_assertion(self):
        result=verify(self.original,guarded(self.original))
        self.assertTrue(result['original_assertions_helpers_and_data_exact'])
        self.assertEqual(result['guards'],9)

    def test_removed_assertion_is_rejected(self):
        actual=guarded(self.original).replace('Assert.True(Fma.IsSupported);','',1)
        with self.assertRaises(AssertionError):verify(self.original,actual)

    def test_guard_after_test_work_is_rejected(self):
        actual=guarded(self.original).replace('        '+FMA,'        var unapproved = new byte[100];\n        '+FMA,1)
        with self.assertRaises(AssertionError):verify(self.original,actual)

    def test_changed_inline_case_is_rejected(self):
        actual=guarded(self.original).replace('[InlineData(7, 33)]','[InlineData(7, 32)]',1)
        with self.assertRaises(AssertionError):verify(self.original,actual)

    def test_actual_qualified_case_names_are_complete(self):
        cases=qualified_census()
        self.assertEqual(len(cases['passed']),40);self.assertEqual(len(cases['skipped']),1)
        self.assertTrue(all('RuntimeIdentityTests' not in n for v in cases.values() for n in v))


if __name__=='__main__':unittest.main()
