"""Retain complete-workload clocks and strict application/ownership checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import audio_protocol as protocol
import audio_worker as audio
import bench


class AudioWorkerTests(unittest.TestCase):
    def fixture(self):
        cases = [dict(name=f'clip-{i}', samples=16000, raw_sha256=str(i)*64, expected={'token_ids': [i]}) for i in range(20)]
        return dict(schema=1, family='parakeet', warmup_passes=1, measured_passes=3, cases=cases)

    def record(self, fixture):
        rows = []
        ticks = 0
        for iteration in range(4):
            for i, case in enumerate(fixture['cases']):
                seconds = (i+1)*(iteration+1)
                rows.append(dict(name=case['name'], **{'pass': iteration}, phase='warmup' if iteration == 0 else 'measured',
                                 seconds=seconds, start_ticks=ticks, end_ticks=ticks+seconds, frequency=1,
                                 result=copy.deepcopy(case['expected']), maximum_centroid_error=0., input_sha256=case['raw_sha256'], ownership=True))
                ticks += seconds
        return dict(schema=1, family='parakeet', engine='managed', conformance=False, held_outputs_unchanged=True,
                    affinity=4, setup_seconds=123., flags={}, records=rows)

    def test_full_twenty_clip_passes_exclude_warmup_and_setup(self):
        fixture = self.fixture()
        value = self.record(fixture)
        samples = audio.measured(value, fixture, [c['name'] for c in fixture['cases']], 'timing')
        self.assertEqual([420000., 630000., 840000.], samples)
        self.assertEqual(80, len(value['records']))

    def test_mean_is_preserved_in_shared_summary(self):
        workers = [dict(case='parakeet', pair=pair, role=role, engine=engine, aggregation='mean', samples_ms=[1., 1., 100.])
                   for pair in (1, 2) for role in ('L0', 'L1') for engine in ('lok', 'ort')]
        summary = bench.summarize(workers)['parakeet']
        self.assertEqual(34., summary['arms']['L1']['lok_ms'])
        self.assertEqual(1., summary['candidate_control'])

    def test_missing_duplicate_or_reordered_calls_fail(self):
        fixture = self.fixture()
        for kind in ('missing', 'duplicate', 'reordered'):
            value = self.record(fixture)
            if kind == 'missing':
                value['records'].pop()
            elif kind == 'duplicate':
                value['records'][20] = value['records'][21]
            else:
                value['records'][20:22] = reversed(value['records'][20:22])
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                audio.measured(value, fixture, [c['name'] for c in fixture['cases']], 'timing')

    def test_decisions_input_identity_and_ownership_are_required(self):
        fixture = self.fixture()
        for kind in ('decision', 'input', 'ownership', 'held', 'clock'):
            value = self.record(fixture)
            row = value['records'][20]
            if kind == 'decision': row['result']['token_ids'][0] += 1
            elif kind == 'input': row['input_sha256'] = 'f'*64
            elif kind == 'ownership': row['ownership'] = False
            elif kind == 'held': value['held_outputs_unchanged'] = False
            else: row['end_ticks'] += 1
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                audio.measured(value, fixture, [c['name'] for c in fixture['cases']], 'timing')

    def test_centroid_bound_and_exact_integer_decisions(self):
        self.assertLess(protocol.check_result({'speakers': [{'centroid': [1.00005]}]}, {'speakers': [{'centroid': [1.]}]}, family='pyannote'), 1e-4)
        with self.assertRaises(AssertionError):
            protocol.check_result({'speakers': [{'centroid': [1.0002]}]}, {'speakers': [{'centroid': [1.]}]}, family='pyannote')
        with self.assertRaises(AssertionError):
            protocol.check_result({'token_ids': [1.]}, {'token_ids': [1]}, family='parakeet')
        with self.assertRaises(AssertionError):
            protocol.check_result({'speakers': [{'centroid': [float('nan')]}]}, {'speakers': [{'centroid': [1.]}]}, family='pyannote')

    def test_reduced_workload_is_rejected(self):
        fixture = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'fixture.json'
            path.write_text(json.dumps(fixture))
            self.assertEqual(20, len(audio.manifest(path, 'parakeet')[1]))
            fixture['cases'].pop()
            path.write_text(json.dumps(fixture))
            with self.assertRaises(Exception):
                audio.manifest(path, 'parakeet')


if __name__ == '__main__':
    unittest.main()
