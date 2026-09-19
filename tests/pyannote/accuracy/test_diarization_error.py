"""Exact labeled examples test the scoring policy, not model output agreement."""
import unittest

from diarization_error import score


class ScoringTests(unittest.TestCase):
    def test_exact_components(self):
        # Expected: total, correct, missed, false alarm, confused, DER.
        cases = [
            ('renamed', [(0, 1, 'A'), (1, 2, 'B')], [(0, 1, 7), (1, 2, 3)], (2, 2, 0, 0, 0, 0)),
            ('miss', [(0, 2, 'A')], [(0, 1, 7)], (2, 1, 1, 0, 0, .5)),
            ('false alarm', [(0, 1, 'A')], [(0, 2, 7)], (1, 1, 0, 1, 0, 1)),
            ('merged speakers', [(0, 1, 'A'), (1, 2, 'B')], [(0, 2, 7)], (2, 1, 0, 0, 1, .5)),
            ('overlap miss', [(0, 2, 'A'), (0, 2, 'B')], [(0, 2, 7)], (4, 2, 2, 0, 0, .5)),
            ('overlap exact', [(0, 2, 'A'), (0, 2, 'B')], [(0, 2, 7), (0, 2, 3)], (4, 4, 0, 0, 0, 0)),
            ('silent', [], [], (0, 0, 0, 0, 0, 0)),
            ('silent false alarm', [], [(0, 1, 7)], (0, 0, 0, 1, 0, 1)),
            ('reference without output', [(0, 1, 'A')], [], (1, 0, 1, 0, 0, 1)),
            ('typed labels', [(0, 1, 'A'), (1, 2, 'B')], [(0, 1, 1), (1, 2, '1')], (2, 2, 0, 0, 0, 0)),
        ]
        keys = ('reference_speaker_seconds', 'correct_speaker_seconds', 'missed_speaker_seconds',
                'false_alarm_speaker_seconds', 'confused_speaker_seconds', 'diarization_error_rate')
        for name, reference, hypothesis, expected in cases:
            with self.subTest(name=name):
                result = score(reference, hypothesis, 2)
                for key, value in zip(keys, expected):
                    self.assertAlmostEqual(result[key], value, places=12)
                self.assertEqual(result['collar_seconds'], 0)
                self.assertTrue(result['overlap_included'])

    def test_no_boundary_collar(self):
        result = score([(0, 1, 'A')], [(.1, 1.1, 7)], 2)
        self.assertAlmostEqual(result['missed_speaker_seconds'], .1)
        self.assertAlmostEqual(result['false_alarm_speaker_seconds'], .1)
        self.assertAlmostEqual(result['diarization_error_rate'], .2)

    def test_refuses_malformed_intervals(self):
        for intervals in ([(-1, 1, 0)], [(0, 3, 0)], [(0, float('nan'), 0)], [(1, 1, 0)],
                          [(0, 1, True)], [(0, 1, 1.5)], [(0, 1, 0), (.5, 2, 0)], [(0, 1)]):
            with self.subTest(intervals=intervals), self.assertRaises(ValueError):
                score(intervals, [], 2)
        for duration in (0, -1, float('inf'), float('nan'), True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                score([], [], duration)


if __name__ == '__main__':
    unittest.main()
