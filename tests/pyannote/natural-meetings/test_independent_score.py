import unittest
from independent_score import score


class IndependentScoreTests(unittest.TestCase):
    def test_perfect_relabeling_and_extra_speaker(self):
        reference = [(0, 3, 'A'), (2, 4, 'B')]
        result = score(reference, [(0, 3, 9), (2, 4, 5)], 5)
        self.assertEqual(result['reference_speaker_seconds'], 5)
        self.assertEqual(result['diarization_error_rate'], 0)
        result = score(reference, [(0, 3, 9), (2, 4, 5), (4, 5, 8)], 5)
        self.assertEqual(result['false_alarm_speaker_seconds'], 1)
        self.assertEqual(result['diarization_error_rate'], .2)

    def test_overlap_and_merged_speakers(self):
        result = score([(0, 3, 'A'), (2, 4, 'B')], [(0, 4, 'X')], 4)
        self.assertEqual(result['missed_speaker_seconds'], 1)
        self.assertEqual(result['confused_speaker_seconds'], 1)
        self.assertEqual(result['diarization_error_rate'], .4)

    def test_silent_reference_and_invalid_intervals(self):
        self.assertEqual(score([], [], 2)['diarization_error_rate'], 0)
        self.assertEqual(score([], [(0, 1, 'X')], 2)['diarization_error_rate'], 1)
        self.assertEqual(score([(1, 2, 'A'), (0, 1, 'A')], [(0, 2, 'B')], 2)['diarization_error_rate'], 0)
        for rows in [[(-1, 1, 'A')], [(0, 2, 'A'), (1, 2, 'A')]]:
            with self.assertRaises(AssertionError):
                score(rows, [], 2)


if __name__ == '__main__':
    unittest.main()
