"""Retain original invariants and require independent emitter/union accounting."""
import io
import unittest
import test_analysis as original
import analyze_v2 as successor
from common import *

# Original tests did not need an emitter field. Add one without changing their
# cases or expected refusal/results, then run them against the successor.
old_suspension = original.suspension
def suspension():
    return [dict(event, thread=2) for event in old_suspension()]
original.suspension = suspension
original.pair_pauses = successor.pair_pauses


class SuccessorTests(original.AnalysisTests):
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
    assert not (BASE / 'unit-tests-v2.json').exists()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SuccessorTests))
    own = psutil.Process()
    save(BASE / 'unit-tests-v2.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        errors=len(result.errors), failures=len(result.failures), skips=len(result.skipped), output=stream.getvalue(),
        identity=dict(pid=own.pid, birth=own.create_time()), analysis=pin(TOOLS / 'analyze_v2.py'), tests_source=pin(Path(__file__))))
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
