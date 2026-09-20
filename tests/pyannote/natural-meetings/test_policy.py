import unittest
from common import annotations, coverage, select


class PolicyTests(unittest.TestCase):
    def test_selection_is_fixed_and_refuses_changed_metadata(self):
        self.assertEqual(select('IS1009b ES2005a IS1009a ES2004a'), ['ES2004a', 'IS1009a'])
        for text in ['ES2004a', 'ES2004a IS1009a ES2004a', 'ES2003a ES2004a IS1009a']:
            with self.assertRaises(ValueError):
                select(text)

    def test_decimal_union_clipping_and_overlap(self):
        text = '\n'.join('SPEAKER ES2004a 1 ' + row + ' <NA> <NA> ' + speaker + ' <NA> <NA>' for row, speaker in [('0.1 0.2', 'A'), ('0.3 0.7', 'A'), ('0.5 0.7', 'B'), ('599.5 2', 'A'), ('601 1', 'C')])
        labels = annotations(text, 'ES2004a 1 0 700', 'ES2004a')
        self.assertEqual(labels, [(0.1, 1., 'A'), (.5, 1.2, 'B'), (599.5, 600., 'A')])
        counts = coverage(labels)
        self.assertAlmostEqual(counts['reference_speaker_seconds'], 2.1)
        self.assertAlmostEqual(counts['overlap_seconds'], .5)
        self.assertEqual(counts['speakers'], 2)

    def test_invalid_intervals_and_uem_are_refused(self):
        valid = 'SPEAKER ES2004a 1 0 1 <NA> <NA> A <NA> <NA>'
        for rttm in [valid.replace('0 1 <NA>', '-1 1 <NA>'), valid.replace('0 1 <NA>', 'NaN 1 <NA>'), valid.replace('0 1 <NA>', '0 0 <NA>'), valid.replace('ES2004a', 'IS1009a'), valid + ' extra']:
            with self.assertRaises(ValueError):
                annotations(rttm, 'ES2004a 1 0 600', 'ES2004a')
        for uem in ['ES2004a 1 1 600', 'ES2004a 1 0 599', 'ES2004a 1 0 NaN']:
            with self.assertRaises(ValueError):
                annotations(valid, uem, 'ES2004a')


if __name__ == '__main__':
    unittest.main()
