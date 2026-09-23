import base64
import struct
import unittest
from audit import CUSTOM, CLR, reconcile
from census import census


def fixture():
    rows = []; events = [dict(index=0, provider=CLR, name='test', id=1, pid=7, thread=8, ms=0, rawBase64='', rawLength=0)]
    for case_index, case in enumerate(census()['cases']):
        row = dict(case, index=case_index, exact=True, inputs=True, ownership=True, setupTicks=1, output='a' * 64, clocks=[])
        rows.append(row)
        for i in range(780):
            counter = 100 * (case_index * 780 + i) + 1
            row['clocks'].append(dict(iteration=i, warmup=i < 600, marker=counter, start=counter + 1, stop=counter + 11, ticks=10,
                allocated=i, allocatedAfter=i + 1, totalAllocated=i, totalAllocatedAfter=i + 1,
                gc0=0, gc1=0, gc2=0, after0=0, after1=0, after2=0))
            for event_id, count in [(1, counter), (2, counter + 11)]:
                raw = struct.pack('<iiq', case_index, i, count)
                events.append(dict(index=len(events), provider=CUSTOM, name=str(event_id), id=event_id, pid=7, thread=8, ms=count,
                    rawBase64=base64.b64encode(raw).decode(), rawLength=16,
                    payload=dict(fixture=str(case_index), iteration=str(i), counter=str(count))))
    return dict(passed=True, diagnosticOnly=True, protocol='parakeet-pad-runtime-diagnostic-v1', calls=9360,
        warmups=7200, measured=2160, frequency=1000, rows=rows, pid=7, nativeThread=8), events, dict(
        complete=True, lost=0, clr_events=1, protocol='all-event-records-v1', recorded=len(events),
        allCounts={CLR + ':test': 1, CUSTOM + ':1': 9360, CUSTOM + ':2': 9360})


class AuditTests(unittest.TestCase):
    def test_complete_calls_and_blocks(self):
        value, events, summary = fixture(); result = reconcile(value, events, summary)
        self.assertEqual((len(result['calls']), len(result['blocks']), result['markers']), (9360, 156, 18720))
        self.assertEqual((result['blocks'][-1]['case'], result['blocks'][-1]['last']), (11, 779))

    def test_loss_and_missing_marker(self):
        value, events, summary = fixture(); summary['lost'] = 1
        with self.assertRaises(AssertionError): reconcile(value, events, summary)
        summary['lost'] = 0
        with self.assertRaises(AssertionError): reconcile(value, events[:-1], summary)

    def test_wrong_thread_case_and_counter(self):
        value, events, summary = fixture(); events[12]['thread'] = 9
        with self.assertRaises(AssertionError): reconcile(value, events, summary)
        events[12]['thread'] = 8
        events[12]['rawBase64'] = base64.b64encode(struct.pack('<iiq', 1, 5, 512)).decode()
        with self.assertRaises(AssertionError): reconcile(value, events, summary)
        events[12]['rawBase64'] = base64.b64encode(struct.pack('<iiq', 0, 5, 999)).decode()
        with self.assertRaises(AssertionError): reconcile(value, events, summary)

    def test_clock_collection_and_ownership(self):
        value, events, summary = fixture(); clock = value['rows'][1]['clocks'][70]; clock['ticks'] = 11
        with self.assertRaises(AssertionError): reconcile(value, events, summary)
        clock['ticks'] = 10; clock['after2'] = -1
        with self.assertRaises(AssertionError): reconcile(value, events, summary)
        clock['after2'] = 0; value['rows'][2]['ownership'] = False
        with self.assertRaises(AssertionError): reconcile(value, events, summary)


if __name__ == '__main__': unittest.main()
