import copy
import unittest
from audit import validate_records,validate_decision
from scoring import metrics,total


class RecordTests(unittest.TestCase):
    def test_scores_use_corpus_denominators_and_independent_edits(self):
        value=total([metrics('A B C D','A B C X'),metrics('E','')])
        self.assertEqual(value['word_errors'],2);self.assertEqual(value['reference_words'],5)
        self.assertEqual(value['word_error_rate'],.4)
        self.assertEqual(metrics('L’été à Paris','lʼété à paris')['word_errors'],0)
        self.assertEqual(metrics('café','cafe')['character_errors'],1)
        with self.assertRaises(ValueError):metrics('…','text')

    def test_bad_clocks_ownership_repeat_and_tokens_are_refused(self):
        cases=[dict(name=str(i),language='fr',pcm_sha256='a'*64) for i in range(40)]
        decision=dict(text='bonjour',token_ids=[123,50257],stop_reason='EndToken',skipped_as_no_speech=False)
        rows=[dict(**c,repeat=i==40,decision=copy.deepcopy(decision),seconds=.5,start_ticks=10*i,end_ticks=10*i+5,
                   frequency=10,input_and_held_results_unchanged=True) for i,c in enumerate(cases+cases[:1])]
        validate_records('whisper','ort',rows,cases)
        mutations=[lambda r:r.pop(),lambda r:r[0].__setitem__('seconds',.2),lambda r:r[0].__setitem__('frequency',0),
                   lambda r:r[0].__setitem__('input_and_held_results_unchanged',False),lambda r:r[-1]['decision'].__setitem__('text','changed'),
                   lambda r:r[0]['decision'].__setitem__('token_ids',[True,50257]),lambda r:r[0]['decision'].__setitem__('token_ids',[123]),
                   lambda r:r[0].__setitem__('repeat',True)]
        for mutate in mutations:
            damaged=copy.deepcopy(rows);mutate(damaged)
            with self.assertRaises(AssertionError):validate_records('whisper','ort',damaged,cases)
        with self.assertRaises(AssertionError):validate_decision('parakeet',dict(text='x',token_ids=[8192],frame_indices=[0],duration_frames=[1],encoded_frames=1,decoder_calls=1,stop_reason='EndOfAudio'))


if __name__=='__main__':unittest.main()
