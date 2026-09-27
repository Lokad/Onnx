"""The reused scorer must reproduce every closed case and reject incomplete clocks."""
import unittest
from prepare import GRAPH, read
from protocol import CASES, ORDER
from statistics import summarize


class Scores(unittest.TestCase):
    def reports(self, key):
        return {role: read(GRAPH/'collected'/('timing-'+key+'-'+role)/'output/result.json') for role in ORDER}

    def test_all_eight_closed_results_are_preserved(self):
        rows = read(GRAPH/'analysis.json')['performance']
        for key in CASES:
            with self.subTest(key=key):
                expected = dict(next(r for r in rows if r['key'] == key)); expected.pop('key')
                self.assertEqual(summarize(self.reports(key)), expected)

    def test_last_clock_cannot_disappear(self):
        reports = self.reports('e5-8tok'); reports['ort-a']['clocks'].pop()
        with self.assertRaises(AssertionError): summarize(reports)

    def test_mixed_case_is_rejected(self):
        reports = self.reports('e5-30tok'); reports['candidate-b']['key'] = 'resnet50'
        with self.assertRaises(AssertionError): summarize(reports)


if __name__ == '__main__': unittest.main()
