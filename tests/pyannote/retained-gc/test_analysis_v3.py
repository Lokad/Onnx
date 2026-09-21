"""Retain original invariants and require independent emitter/union accounting."""
import io
import unittest
import test_analysis as original
import analyze_v3 as successor
from common import *

# Original tests did not need an emitter field. Add one without changing their
# cases or expected refusal/results, then run them against the successor.
old_suspension = original.suspension
def suspension():
    return [dict(event, thread=2) for event in old_suspension()]
original.suspension = suspension
original.pair_pauses = successor.pair_pauses


class SuccessorTests(original.AnalysisTests):
    def test_crossing_background_gc_is_counted_once_at_start(self):
        gcs = [dict(count=1, depth=2, start_ms=8, end_ms=12), dict(count=2, depth=0, start_ms=13, end_ms=14)]
        self.assertEqual(successor.generation_counts(gcs, 0, 10, 'start_ms'), [1, 1, 1])
        self.assertEqual(successor.generation_counts(gcs, 10, 20, 'start_ms'), [1, 0, 0])
        self.assertEqual(successor.generation_counts(gcs, 10, 20, 'end_ms'), [2, 1, 1])

    def test_gc_duration_is_not_suspension_duration(self):
        gcs = [dict(count=1, depth=2, start_ms=1, end_ms=20)]
        self.assertEqual(successor.generation_counts(gcs, 0, 25, 'start_ms'), [1, 1, 1])
        self.assertEqual(successor.union_length([(1, 2), (18, 20)], 0, 25), 3)

    def test_distinct_emitters_can_overlap_resume_envelopes(self):
        a = suspension()
        b = [dict(event, index=event['index'] + 4, thread=3, ms=event['ms'] + 1.5) for event in suspension()]
        events = sorted(a + b, key=lambda event: event['ms'])
        pauses = successor.pair_pauses(events)
        self.assertEqual(len(pauses), 2)
        self.assertEqual({row['thread'] for row in pauses}, {2, 3})
        self.assertEqual(successor.union_length([(row['start_ms'], row['end_ms']) for row in pauses], 0, 10), 4.5)

    def test_overlap_does_not_allow_a_missing_emitter_end(self):
        a = suspension()
        b = [dict(event, index=event['index'] + 4, thread=3, ms=event['ms'] + 1.5) for event in suspension()]
        with self.assertRaises(AssertionError):
            successor.pair_pauses(sorted(a[:-1] + b, key=lambda event: event['ms']))


if __name__ == '__main__':
    assert not (BASE / 'unit-tests-v3.json').exists()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SuccessorTests))
    own = psutil.Process()
    save(BASE / 'unit-tests-v3.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        errors=len(result.errors), failures=len(result.failures), skips=len(result.skipped), output=stream.getvalue(),
        identity=dict(pid=own.pid, birth=own.create_time()), analysis=pin(TOOLS / 'analyze_v3.py'), tests_source=pin(Path(__file__))))
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
