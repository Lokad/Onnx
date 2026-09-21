"""Independent invariants for event pairing and interval accounting."""
import io
import unittest
from analyze import pair_pauses, pair_requests, union_length, collections_from, STAGES
from common import *


def suspension():
    return [dict(index=i, name=name, pid=1, ms=float(i + 1), payload=dict(ClrInstanceID='7', Reason='SuspendForGC', Count='3'))
        for i, name in enumerate(STAGES)]


def boundary(phase, name, at):
    return dict(index=int(at), provider='Lokad-Pyannote-Diagnostic', name='Boundary', id=1, pid=1, thread=2, ms=at,
        payload=dict(phase=phase, name=name, **{'pass': '1'}))


def collection(phase, count, depth, at):
    return dict(index=int(at), name='GC/' + phase, pid=1, ms=at,
        payload=dict(ClrInstanceID='7', Count=str(count), Depth=str(depth), Type='BackgroundGC', Reason='AllocLarge'))


class AnalysisTests(unittest.TestCase):
    def test_interval_clips_both_boundaries(self):
        self.assertEqual(union_length([(-5, 2), (4, 20)], 0, 7), 5)

    def test_overlapping_and_nested_intervals_do_not_double_count(self):
        self.assertEqual(union_length([(1, 6), (2, 3), (5, 9), (1, 6)], 0, 10), 8)

    def test_partition_conserves_interval_measure(self):
        spans = [(-2, 3), (2, 5), (7, 12)]
        self.assertEqual(union_length(spans, 0, 10), union_length(spans, 0, 4) + union_length(spans, 4, 10))

    def test_reversed_interval_is_refused(self):
        with self.assertRaises(AssertionError):
            union_length([(3, 2)], 0, 5)

    def test_complete_suspension_preserves_components(self):
        row, = pair_pauses(suspension())
        self.assertEqual([row[key] for key in ['start_ms', 'suspended_ms', 'restart_ms', 'end_ms']], [1, 2, 3, 4])
        self.assertEqual(row['reason'], 'SuspendForGC')

    def test_missing_suspend_phase_is_refused(self):
        with self.assertRaises(AssertionError):
            pair_pauses(suspension()[::2])

    def test_nested_suspension_is_refused(self):
        values = suspension()
        with self.assertRaises(AssertionError):
            pair_pauses([values[0], dict(values[0], ms=1.5), *values[1:]])

    def test_unsorted_event_stream_is_refused(self):
        values = suspension()
        values[2]['ms'] = .5
        with self.assertRaises(AssertionError):
            pair_pauses(values)

    def test_missing_request_end_is_refused(self):
        with self.assertRaises(AssertionError):
            pair_requests([boundary('begin', 'full', 1)])

    def test_wrong_request_end_is_refused(self):
        with self.assertRaises(AssertionError):
            pair_requests([boundary('begin', 'full', 1), boundary('end', 'crop', 2)])

    def test_complete_requests_preserve_identity_and_boundaries(self):
        rows = pair_requests([boundary('begin', 'full', 1), boundary('end', 'full', 9),
            boundary('begin', 'crop', 10), boundary('end', 'crop', 11)])
        self.assertEqual([(r['name'], r['start_ms'], r['end_ms']) for r in rows], [('full', 1, 9), ('crop', 10, 11)])

    def test_background_gc_can_overlap_a_lower_generation_collection(self):
        rows = collections_from([collection('Start', 1, 2, 1), collection('Start', 2, 0, 2),
            collection('Stop', 2, 0, 3), collection('Stop', 1, 2, 4)])
        self.assertEqual([(r['count'], r['depth']) for r in rows], [(2, 0), (1, 2)])

    def test_missing_collection_stop_is_refused(self):
        with self.assertRaises(AssertionError):
            collections_from([collection('Start', 1, 2, 1)])


if __name__ == '__main__':
    assert not (BASE / 'unit-tests.json').exists()
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AnalysisTests))
    own = psutil.Process()
    save(BASE / 'unit-tests.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
        errors=len(result.errors), failures=len(result.failures), skips=len(result.skipped), output=stream.getvalue(),
        identity=dict(pid=own.pid, birth=own.create_time()), analysis=pin(TOOLS / 'analyze.py'), tests_source=pin(Path(__file__))))
    print(stream.getvalue())
    raise SystemExit(0 if result.wasSuccessful() else 1)
