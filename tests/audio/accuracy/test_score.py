import copy
import unittest
from score import decision, distance, metrics, normalize, validate_coverage, validate_selection


class ScoringTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(normalize("  'Don't' stop—now! １２ ﬁancée Straße "), "DON'T STOP NOW 12 FIANCÉE STRASSE")
        self.assertEqual(normalize("a''b a’b x_y"), "A B A B X Y")

    def test_hand_checked_errors(self):
        for reference, hypothesis, expected in (
            ('A B C', 'A X C', (1, 0, 0)), ('A B C', 'A C', (0, 1, 0)),
            ('A B', 'A B C', (0, 0, 1)), ('A B', '', (0, 2, 0)),
            ("Don't stop!", "DON'T STOP", (0, 0, 0)), ('A B C', 'A X C D', (1, 0, 1))):
            row = metrics(reference, hypothesis)
            self.assertEqual(tuple(row[k] for k in ('substitutions', 'deletions', 'insertions')), expected)
            self.assertEqual(row['word_errors'], sum(expected))
        self.assertEqual(distance('KITTEN', 'SITTING'), 3)
        self.assertEqual(distance('', 'ABC'), 3)

    def test_empty_reference_refused(self):
        with self.assertRaisesRegex(ValueError, 'no scored characters'):
            metrics('!?', 'words')

    def test_ambiguous_alignment_still_checks_total_distance(self):
        self.assertEqual(metrics('A B A', 'B A B')['word_errors'], 2)

    def test_case_coverage_and_repeat_refusals(self):
        good = [dict(name='a', repeat=False), dict(name='b', repeat=False), dict(name='a', repeat=True)]
        validate_coverage(good, ['a', 'b'], True)
        for bad in (good[:-1], good[::-1], [good[0], good[0], good[2]], [good[0], good[1], dict(name='a', repeat=False)]):
            with self.assertRaises(ValueError):
                validate_coverage(bad, ['a', 'b'], True)

    def test_selection_transcript_duplicate_and_policy_refusals(self):
        pins = dict(rows=22, excluded_speakers=[1272], speakers=10, duration_bands_seconds=[[4, 10], [12, 25]])
        inventory = [dict(id=f'{speaker}-{part}', speaker_id=speaker, chapter_id=3, samples=16000 * duration,
                          text=f'TRANSCRIPT {speaker} {part}', audio_sha256=f'audio-{speaker}-{part}')
                     for speaker in range(1, 12) for part, duration in [('a', 5), ('b', 15)]]
        chosen = inventory[:20]
        selection = dict(dataset=pins, inventory=list(reversed(inventory)), selected=chosen)
        audio = dict(schema=1, sample_rate=16000, dataset=pins,
            cases=[dict(name=r['id'], speaker_id=r['speaker_id'], chapter_id=r['chapter_id'], samples=r['samples'],
                        reference_text=r['text'], flac_sha256=r['audio_sha256'], pcm_sha256=r['id'],
                        language='en', max_new_tokens=444) for r in chosen])
        validate_selection(audio, selection, pins)
        for key, value in [('name', 'changed'), ('reference_text', 'CHANGED'), ('speaker_id', 12),
                           ('samples', 1), ('pcm_sha256', audio['cases'][1]['pcm_sha256']), ('max_new_tokens', 3)]:
            bad = copy.deepcopy(audio)
            bad['cases'][0][key] = value
            with self.assertRaises(ValueError):
                validate_selection(bad, selection, pins)
        changed = copy.deepcopy(selection)
        changed['selected'][-1] = inventory[-1]
        with self.assertRaises(ValueError):
            validate_selection(audio, changed, pins)
        changed = copy.deepcopy(selection)
        changed['inventory'][0]['id'] = changed['inventory'][1]['id']
        with self.assertRaises(ValueError):
            validate_selection(audio, changed, pins)

    def test_application_metadata_is_part_of_agreement(self):
        native = dict(text='text', tokens=[4, 50257], stop_reason='EndToken', skipped_as_no_speech=False)
        managed = dict(Text='text', TokenIds=[4, 50257], StopReason='EndToken', SkippedAsNoSpeech=False)
        self.assertEqual(decision('whisper', native, True), decision('whisper', managed))
        managed['StopReason'] = 'TokenLimit'
        self.assertNotEqual(decision('whisper', native, True), decision('whisper', managed))


if __name__ == '__main__':
    unittest.main()
