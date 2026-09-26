"""Reject damaged priming evidence, using complete retained calls as fixtures."""
import base64
import copy
import json
import struct
import unittest
from audit import prefix_checked
from prepare import EVIDENCE


class PrefixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original = json.loads((EVIDENCE / 'collected/current-capture/result.json').read_text())
        first = original['rows'][0]['clocks'][0]['marker']
        last = original['rows'][-1]['clocks'][-1]['stop']
        span = last - first + 100
        assert 10 * original['frequency'] < span < 60 * original['frequency']
        passes = []; markers = []
        for round_index in range(2):
            rows = copy.deepcopy(original['rows'])
            delta = round_index * span
            for case, row in enumerate(rows):
                for clock in row['clocks']:
                    for name in ('marker', 'start', 'stop'): clock[name] += delta
                    for identifier, counter in ((3, clock['marker']), (4, clock['stop'])):
                        number = round_index * 780 + clock['iteration']
                        markers.append(dict(provider='Lokad-Parakeet-Pad-Diagnostic', id=identifier,
                            pid=original['pid'], thread=original['nativeThread'],
                            ms=(counter - first) * 1000 / original['frequency'],
                            rawBase64=base64.b64encode(struct.pack('<iiq', case, number, counter)).decode(),
                            payload=dict(fixture=case, iteration=number, counter=counter)))
            passes.append(dict(round=round_index, start=first + delta - 1, end=last + delta + 1, rows=rows))
        value = copy.deepcopy(original)
        for row in value['rows']:
            for clock in row['clocks']:
                for name in ('marker', 'start', 'stop'): clock[name] += 2 * span
        markers.append(dict(provider='Lokad-Parakeet-Pad-Diagnostic', id=1,
                            ms=2 * span * 1000 / original['frequency']))
        prefix = dict(passed=True, protocol='pad-census-ten-seconds-after-first-v1',
            pid=original['pid'], nativeThread=original['nativeThread'], frequency=original['frequency'],
            began=first - 2, firstEnd=passes[0]['end'], ended=passes[-1]['end'], calls=18720, passes=passes)
        cls.prefix, cls.value, cls.events = prefix, value, markers

    def test_complete_retained_calls_reconcile(self):
        self.assertEqual(prefix_checked(self.prefix, self.value, self.events)['calls'], 18720)

    def test_missing_marker_rejected(self):
        with self.assertRaises(AssertionError):
            prefix_checked(self.prefix, self.value, self.events[1:])

    def test_wrong_round_counter_rejected(self):
        events = list(self.events)
        events[18720] = copy.deepcopy(events[18720])
        events[18720]['payload']['iteration'] = 0
        with self.assertRaises(AssertionError): prefix_checked(self.prefix, self.value, events)

    def test_short_warmup_rejected(self):
        prefix = dict(self.prefix, frequency=self.prefix['frequency'] * 4)
        value = dict(self.value, frequency=prefix['frequency'])
        with self.assertRaises(AssertionError): prefix_checked(prefix, value, self.events)

    def test_suffix_inside_prefix_rejected(self):
        value = copy.deepcopy(self.value)
        value['rows'][0]['clocks'][0]['marker'] = self.prefix['ended'] - 1
        with self.assertRaises(AssertionError): prefix_checked(self.prefix, value, self.events)


if __name__ == '__main__': unittest.main()
