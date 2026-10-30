"""Retain complete-workload clocks and strict application/ownership checks."""
import copy
import json
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest import mock

import audio_protocol as protocol
import audio_worker as audio
import bench


class AudioWorkerTests(unittest.TestCase):
    def test_expired_full_budget_never_launches_a_worker(self):
        with mock.patch.object(audio.time, 'monotonic', return_value=101), mock.patch.object(audio.common, 'spawn_child') as spawn:
            with self.assertRaisesRegex(ValueError, 'duration limit exceeded before launch'):
                audio.child({}, Path('.'), Path('.'), 'L1', 1, 'lok', 'parakeet', Path('fixture.json'), {}, [],
                            'timing', 'dotnet', 'python', 1, overall_deadline=100)
            spawn.assert_not_called()

    def test_full_budget_expiry_stops_only_owned_worker_and_keeps_failure_receipt(self):
        class OwnedProcess:
            pid = 4242
            returncode = None
            terminated = False

            def poll(self): return self.returncode
            def terminate(self): self.terminated = True; self.returncode = -15
            def wait(self, timeout=None): return self.returncode
            def kill(self): raise AssertionError('graceful termination should suffice')

        process = OwnedProcess()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with mock.patch.dict(sys.modules, {'psutil': types.SimpleNamespace()}), \
                    mock.patch.object(audio.time, 'monotonic', side_effect=[0, 0, 101]), \
                    mock.patch.object(audio.common, 'spawn_child', return_value=process) as spawn:
                with self.assertRaisesRegex(ValueError, 'full benchmark duration limit exceeded'):
                    audio.child({'runner': Path('caller.dll')}, output, output, 'L1', 1, 'lok', 'parakeet',
                                output/'fixture.json', self.fixture_v2(), [], 'timing', 'dotnet', 'python', 1,
                                overall_deadline=100)
            self.assertTrue(process.terminated)
            spawn.assert_called_once()
            receipt = json.loads((output/'audio-001-parakeet-l1-lok-timing.supervision.json').read_text())
            self.assertEqual(4242, receipt['pid'])
            self.assertEqual(-15, receipt['exit_code'])
            self.assertEqual('full benchmark duration limit exceeded', receipt['failure'])

    def fixture(self):
        cases = [dict(name=f'clip-{i}', samples=16000, raw_sha256=str(i)*64, expected={'token_ids': [i]}) for i in range(20)]
        return dict(schema=1, family='parakeet', warmup_passes=1, measured_passes=3, cases=cases)

    def record(self, fixture):
        rows = []
        v2 = fixture['schema'] == 2
        phases = (['first-use'] + ['conditioning'] * fixture['conditioning_passes'] + ['measured'] * fixture['measured_passes']) if v2 else ['warmup', 'measured', 'measured', 'measured']
        ticks = 124 if v2 else 0
        for iteration, phase in enumerate(phases):
            for i, case in enumerate(fixture['cases']):
                seconds = (i+1)*(iteration+1)
                rows.append(dict(name=case['name'], **{'pass': iteration}, phase=phase,
                                 seconds=seconds, start_ticks=ticks, end_ticks=ticks+seconds, frequency=1,
                                 result=copy.deepcopy(case['expected']), maximum_centroid_error=0., input_sha256=case['raw_sha256'], ownership=True,
                                 gc_before=[0, 0, 0], gc_after=[1, 0, 0], allocated_bytes=100,
                                 heap_bytes_before=1000, heap_bytes_after=1100, gc_pause_seconds=.1))
                ticks += seconds
        value = dict(schema=fixture['schema'], family='parakeet', engine='managed', conformance=False, held_outputs_unchanged=True,
                     affinity=4, setup_seconds=123., flags={}, records=rows)
        if v2:
            value.update(producer='audio-sustained-v2',
                         schedule=dict(first_use_passes=1, conditioning_passes=fixture['conditioning_passes'], measured_passes=fixture['measured_passes']),
                         startup=dict(entry_utc='2026-10-29T11:00:00Z', entry_ticks=0, setup_start_ticks=1, setup_end_ticks=124, frequency=1, preparation_seconds=1.))
        return value

    def fixture_v2(self):
        fixture = self.fixture()
        fixture.pop('warmup_passes')
        fixture.update(schema=2, protocol='audio-sustained-v2', conditioning_passes=2, measured_passes=4, process_timeout_seconds=1800)
        return fixture

    def test_full_twenty_clip_passes_exclude_warmup_and_setup(self):
        fixture = self.fixture()
        value = self.record(fixture)
        samples = audio.measured(value, fixture, [c['name'] for c in fixture['cases']], 'timing')
        self.assertEqual([420000., 630000., 840000.], samples)
        self.assertEqual(80, len(value['records']))

    def test_v2_keeps_first_use_conditioning_and_setup_separate(self):
        fixture = self.fixture_v2()
        value = self.record(fixture)
        samples = audio.measured(value, fixture, [c['name'] for c in fixture['cases']], 'timing')
        self.assertEqual([840000., 1050000., 1260000., 1470000.], samples)
        self.assertEqual(140, len(value['records']))
        self.assertEqual(['first-use', 'conditioning', 'conditioning', 'measured', 'measured', 'measured', 'measured'], protocol.schedule(fixture, 'timing'))
        # Conformance still exercises every case, without claiming any timing.
        value['records'] = value['records'][:20]
        value['conformance'] = True
        self.assertEqual([], audio.measured(value, fixture, [c['name'] for c in fixture['cases']], 'conformance'))

    def test_v2_cannot_hide_missing_calls_or_mislabel_first_use_as_measured(self):
        fixture = self.fixture_v2()
        for kind in ('missing', 'extra', 'phase', 'pass', 'schedule', 'producer', 'startup', 'pause', 'collections', 'allocation'):
            value = self.record(fixture)
            if kind == 'missing': value['records'].pop()
            elif kind == 'extra': value['records'].append(value['records'][-1])
            elif kind == 'phase': value['records'][0]['phase'] = 'measured'
            elif kind == 'pass': value['records'][60]['pass'] = 2
            elif kind == 'schedule': value['schedule']['conditioning_passes'] = 1
            elif kind == 'producer': value['producer'] = 'audio-full-api-v1'
            elif kind == 'startup': value['startup']['setup_end_ticks'] += 1
            elif kind == 'pause': value['records'][0]['gc_pause_seconds'] = -1
            elif kind == 'collections': value['records'][0]['gc_after'] = [-1, 0, 0]
            else: value['records'][0]['allocated_bytes'] = -1
            with self.subTest(kind=kind), self.assertRaises(AssertionError):
                audio.measured(value, fixture, [c['name'] for c in fixture['cases']], 'timing')

    def test_protocol_versions_require_explicit_bounded_schedules(self):
        for field, bad in [('schema', 3), ('protocol', 'audio-full-api-v1'), ('conditioning_passes', -1),
                           ('conditioning_passes', True), ('measured_passes', 1), ('measured_passes', 33),
                           ('process_timeout_seconds', 0), ('warmup_passes', 1)]:
            fixture = self.fixture_v2()
            fixture[field] = bad
            with self.subTest(field=field, value=bad), self.assertRaises(AssertionError):
                protocol.definition(fixture)
        legacy = self.fixture()
        legacy['warmup_passes'] = 2
        with self.assertRaises(AssertionError):
            protocol.definition(legacy)
        pilot = self.fixture_v2()
        pilot['conditioning_passes'] = 0
        self.assertEqual(['first-use', 'measured', 'measured', 'measured', 'measured'], protocol.schedule(pilot, 'timing'))

    def test_mean_is_preserved_in_shared_summary(self):
        workers = [dict(case='parakeet', pair=pair, role=role, engine=engine, aggregation='mean', samples_ms=[1., 1., 100.])
                   for pair in (1, 2) for role in ('L0', 'L1') for engine in ('lok', 'ort')]
        summary = bench.summarize(workers)['parakeet']
        self.assertEqual(34., summary['arms']['L1']['lok_ms'])
        self.assertEqual([34., 34.], summary['arms']['L1']['lok_process_means_ms'])
        self.assertNotIn('lok_process_medians_ms', summary['arms']['L1'])
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
