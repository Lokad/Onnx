import unittest
from prepare import PRIOR,CONV,MATMUL,changed,edits


class ObservationScope(unittest.TestCase):
    def test_source_changes_reverse_exactly(self):
        for name in [CONV,MATMUL]:
            before=(PRIOR/'source'/name).read_text(encoding='utf8');after=changed(name,before)
            self.assertNotEqual(before,after)
            for old,new in reversed(edits(name)):after=after.replace(new,old)
            self.assertEqual(before,after)

    def test_missing_callsite_is_rejected(self):
        before=(PRIOR/'source'/CONV).read_text(encoding='utf8')
        old,_=edits(CONV)[0]
        with self.assertRaises(AssertionError):changed(CONV,before.replace(old,'missing'))

    def test_duplicate_callsite_is_rejected(self):
        before=(PRIOR/'source'/MATMUL).read_text(encoding='utf8')
        old,_=edits(MATMUL)[0]
        with self.assertRaises(AssertionError):changed(MATMUL,before+'\n'+old)


if __name__=='__main__':unittest.main()
