"""Exercise the reused clock audit against retained events and corrupted evidence."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import unittest
from events import reconcile, CUSTOM, CLR

ROOT = Path(__file__).resolve().parents[3]
PRIOR = ROOT/'artifacts/parakeet-decoder-lstm-runtime-observation-amd-20260927/collected'


def summarize(events, original):
    summary = deepcopy(original)
    for index, event in enumerate(events): event['index'] = index
    summary['recorded'] = len(events)
    summary['allCounts'] = dict(Counter(e['provider']+':'+e['name'] for e in events))
    summary['clr_events'] = sum(e['provider'] == CLR for e in events)
    return summary


class ClockAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.value = json.loads((PRIOR/'trace-capture/output/result.json').read_text())
        cls.value['diagnostics'] = cls.value['diagnostics'][:400]
        all_events = [json.loads(s) for s in (PRIOR/'trace-export/events/events.jsonl').read_text().splitlines()]
        cls.events = [e for e in all_events if e['provider'] != CUSTOM or int(e['payload']['iteration']) < 400]
        cls.summary = summarize(cls.events, json.loads((PRIOR/'trace-export/events/summary.json').read_text()))

    def test_retained_clock_correspondence(self):
        checked = reconcile(self.value, self.events, self.summary)
        self.assertEqual(checked['marker_count'], 800)
        self.assertEqual(len(checked['intervals']), 400)

    def test_rejects_incomplete_or_inconsistent_trace(self):
        bad = deepcopy(self.summary); bad['lost'] = 1
        with self.assertRaises(AssertionError): reconcile(self.value, self.events, bad)
        index = next(i for i,e in enumerate(self.events) if e['provider'] == CUSTOM)
        missing = deepcopy(self.events); missing.pop(index)
        with self.assertRaises(AssertionError): reconcile(self.value, missing, summarize(missing, self.summary))
        for field, value in [('thread', -1), ('ms', self.events[index]['ms']+.1)]:
            bad = deepcopy(self.events); bad[index][field] = value
            with self.assertRaises(AssertionError): reconcile(self.value, bad, self.summary)
        bad = deepcopy(self.events); bad[index]['payload']['counter'] = '0'
        with self.assertRaises(AssertionError): reconcile(self.value, bad, self.summary)


if __name__ == '__main__': unittest.main()
