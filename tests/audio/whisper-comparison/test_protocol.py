import copy
import math
import unittest
from audit import validate_worker, refusal_checks


def fixture(mode):
    cases = [dict(name=name, raw_sha256=name, frontend_sha256='frontend', expected=dict(
        text=name, token_ids=[1, 2, 50257], stop_reason='EndToken', skipped_as_no_speech=False)) for name in ('a', 'b')]
    manifest = dict(family='whisper', cases=cases)
    records = []
    for iteration in range(1 if mode == 'conformance' else 4):
        for case in cases:
            start = 100 * len(records)
            records.append(dict(name=case['name'], **{'pass': iteration}, phase='warmup' if iteration == 0 else 'measured',
                ownership=True, input_sha256=case['raw_sha256'], frontend_sha256=case['frontend_sha256'],
                frequency=1_000_000_000, start_ticks=start, end_ticks=start + 10, seconds=1e-8,
                result=copy.deepcopy(case['expected']), maximum_centroid_error=0.))
    return dict(schema=1, family='whisper', engine='ort', conformance=mode == 'conformance', affinity=4,
        held_outputs_unchanged=True, setup_seconds=1., flags={}, records=records), manifest


class ProtocolTests(unittest.TestCase):
    def test_both_phases_and_existing_nine_refusals(self):
        for mode in ('conformance', 'timing'):
            value, manifest = fixture(mode)
            validate_worker(value, manifest, mode)
            refusal_checks(value, manifest, mode)

    def test_frontend_stop_clock_and_duplicate_refusals(self):
        mutations = [lambda v: v['records'][0].update(frontend_sha256='bad'),
                     lambda v: v['records'][0]['result'].update(stop_reason='TokenLimit'),
                     lambda v: v['records'][0].update(frequency=math.nan),
                     lambda v: v['records'][0].update(frequency=True),
                     lambda v: v['records'][0]['result']['token_ids'].__setitem__(0, True),
                     lambda v: v['records'].__setitem__(1, copy.deepcopy(v['records'][0])),
                     lambda v: v['records'][0].update(end_ticks=1000)]
        for mutation in mutations:
            value, manifest = fixture('timing'); mutation(value)
            with self.assertRaises(AssertionError):
                validate_worker(value, manifest, 'timing')


if __name__ == '__main__': unittest.main()
