"""Reject lost intervals and false coverage instead of merely summing reported phases."""
import copy
import unittest
from reconcile import PHASES, interval, records


def example(eligible=True):
    phase = dict(allocation=1, boundaries=2, validation=1, input_layout=1,
        arithmetic=1, output_epilogue=1, pool_return=1, wrapper=2, fallback=0)
    if not eligible: phase = {k: (10 if k == 'fallback' else 0) for k in PHASES}
    return dict(frequency=100, start=1, stop=11, before=0, after=int(eligible), thread=1,
        eligible=eligible, inner=list(range(2, 11)) if eligible else [], phases=phase)


def coverage():
    calls = [dict(case='crop'+str(crop), index=i, form=i % 15, eligible=i % 9 != 0) for crop in range(3) for i in range(36)]
    expected = {(r['case'], r['index']): dict(production='fixed', values=1) for r in calls}
    rows = []; sequence = 0
    for repeat in range(4):
        for call in calls:
            row = example(call['eligible']); row.update(before=sequence, after=sequence+int(call['eligible']),
                name=call['case'], index=call['index'], form=call['form'], warmup=repeat == 0,
                exact=True, sha256='fixed', values=1)
            row['pass'] = repeat; sequence = row['after']; rows.append(row)
    return rows, calls, expected


class Reconciliation(unittest.TestCase):
    def test_complete_intervals_and_boundary_gaps(self):
        self.assertEqual(sum(interval(example(), 100).values()), 10)
        self.assertEqual(interval(example(False), 100)['fallback'], 10)

    def test_one_tick_loss_is_rejected(self):
        row = example(); row['phases']['boundaries'] -= 1
        with self.assertRaises(AssertionError): interval(row, 100)

    def test_nonmonotonic_clock_is_rejected(self):
        row = example(); row['inner'][3] = 2
        with self.assertRaises(AssertionError): interval(row, 100)

    def test_incorrect_frequency_is_rejected(self):
        with self.assertRaises(AssertionError): interval(example(), 101)
        with self.assertRaises(AssertionError): interval(example(), 0)

    def test_stale_fallback_record_is_rejected(self):
        row = example(False); row['inner'] = list(range(2, 11))
        with self.assertRaises(AssertionError): interval(row, 100)
        row = example(False); row['after'] = 1
        with self.assertRaises(AssertionError): interval(row, 100)

    def test_complete_coverage(self):
        rows, calls, expected = coverage(); actual = records(rows, calls, expected, 100)
        self.assertEqual((actual['eligible'], actual['fallback']), (384, 48))

    def test_missing_and_duplicated_calls_are_rejected(self):
        rows, calls, expected = coverage()
        with self.assertRaises(AssertionError): records(rows[:-1], calls, expected, 100)
        rows[1] = copy.deepcopy(rows[0])
        with self.assertRaises(AssertionError): records(rows, calls, expected, 100)

    def test_output_mismatch_and_sequence_gap_are_rejected(self):
        rows, calls, expected = coverage(); rows[100]['sha256'] = 'other'
        with self.assertRaises(AssertionError): records(rows, calls, expected, 100)
        rows, calls, expected = coverage(); rows[100]['before'] += 1
        with self.assertRaises(AssertionError): records(rows, calls, expected, 100)


if __name__ == '__main__': unittest.main()
