"""Adversarial coverage for launch gates, hardware skips and numerical audits."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
import numpy as np
from candidate_protocol import LIMITS, ROLES, TIMING_ROLES, REQUIRED_TESTS, gate, check_sample, test_results, below, pin, read
import qualify_outputs
import transport

ROOT = Path(__file__).resolve().parents[3]


class ProtocolTests(unittest.TestCase):
    def test_full_suite_prerequisites_build_cli_and_refuse_missing_output(self):
        sys.path.insert(0, str(transport.SITE))
        from supervise import build_prerequisites
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); calls = []
            def worker(name, args, build=False):
                calls.append((name, args)); self.assertTrue(build)
                if name == 'cli-build':
                    output = base/'source/src/Lokad.Onnx.CLI/bin/Release/net10.0'; output.mkdir(parents=True)
                    for filename in ('Lokad.Onnx.CLI.dll', 'Lokad.Onnx.CLI.deps.json', 'Lokad.Onnx.CLI.runtimeconfig.json'):
                        (output/filename).write_text('fixture')
            projects = build_prerequisites(base, worker, ['--tl:off'])
            self.assertEqual([n for n, _ in calls], [n+s for n in ('backend', 'tensors', 'cli', 'il-bridge') for s in ('-restore', '-build')])
            self.assertEqual([n for n, _ in projects[:2]], ['backend', 'tensors'])
            self.assertEqual(calls[5][1][2], base/'source/src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj')
            (base/'source/src/Lokad.Onnx.CLI/bin/Release/net10.0/Lokad.Onnx.CLI.dll').unlink()
            with self.assertRaisesRegex(AssertionError, 'Missing Release CLI'):
                build_prerequisites(base, lambda *args, **kwargs: None, ['--tl:off'])

    def valid_reports(self):
        return {r: dict(pyannote=dict(passed=True, arrays=18, public_calls=16), parakeet=dict(
            audit_consistent=True, application_passed=True, numeric_gate_passed=True, arrays=784, values=3090494)) for r in ROLES}

    def test_native_failure_missing_role_and_truncated_trajectory_stop_timing(self):
        reports = self.valid_reports(); gate(reports)
        for role in ROLES:
            for family, key, value in [('pyannote', 'passed', False), ('pyannote', 'arrays', 17),
                                       ('parakeet', 'numeric_gate_passed', False), ('parakeet', 'values', 3090493)]:
                damaged = copy.deepcopy(reports); damaged[role][family][key] = value
                with self.assertRaises(AssertionError): gate(damaged)
            damaged = copy.deepcopy(reports); del damaged[role]
            with self.assertRaises(AssertionError): gate(damaged)

    def test_resource_limits_and_every_thread_affinity(self):
        sample = dict(seconds=1, available=LIMITS['available'], tmpfs_free=LIMITS['tmpfs_free'], artifact_bytes=1,
                      members=[dict(rss=1, affinity=[2], threads=[dict(affinity=[2])])])
        check_sample(sample)
        for key, value in [('seconds', LIMITS['worker_seconds']), ('available', LIMITS['available']-1),
                           ('tmpfs_free', LIMITS['tmpfs_free']-1), ('artifact_bytes', LIMITS['artifact_bytes']+1)]:
            damaged = copy.deepcopy(sample); damaged[key] = value
            with self.assertRaises(AssertionError): check_sample(damaged)
        damaged = copy.deepcopy(sample); damaged['members'][0]['threads'][0]['affinity'] = [0, 2]
        with self.assertRaises(AssertionError): check_sample(damaged)

    def test_required_avx512_test_skip_or_omission_never_passes(self):
        ns = 'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'
        def fixture(path, outcome, omit=False):
            root = ET.Element('{'+ns+'}TestRun'); results = ET.SubElement(root, '{'+ns+'}Results')
            for name in REQUIRED_TESTS[:-1] if omit else REQUIRED_TESTS:
                ET.SubElement(results, '{'+ns+'}UnitTestResult', testName=name, outcome=outcome)
            summary = ET.SubElement(root, '{'+ns+'}ResultSummary')
            ET.SubElement(summary, '{'+ns+'}Counters', total=str(len(results)), passed=str(len(results) if outcome == 'Passed' else 0))
            ET.ElementTree(root).write(path)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'test.trx'; fixture(path, 'Passed')
            self.assertEqual(test_results(path, 0, REQUIRED_TESTS)['passed'], len(REQUIRED_TESTS))
            for outcome, omitted in [('NotExecuted', False), ('Passed', True), ('Failed', False)]:
                fixture(path, outcome, omitted)
                with self.assertRaises(AssertionError): test_results(path, 0, REQUIRED_TESTS)

    def test_live_e5_blocks_before_any_remote_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'state.json'; path.write_text(json.dumps(dict(complete=False)))
            with patch.object(transport, 'E5_CONTROL', path), patch.object(transport, 'ssh') as ssh:
                with self.assertRaisesRegex(AssertionError, 'still owns'): transport.stage()
                ssh.assert_not_called()

    def test_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(AssertionError): below(Path(tmp), '../escape')

    def test_native_inventory_selects_package_root_not_nested_directory_names(self):
        from prepare_execution import selected_native_files
        retained = read(ROOT/'artifacts/audio-amd-two-family-20260920/collected/frozen.json')
        selected = selected_native_files(retained['external'])
        self.assertEqual(len(selected), 16117)
        self.assertTrue(any('/onnxruntime/transformers/' in name for name in selected))
        self.assertTrue(any('/torch/include/ATen/native/transformers/' in name for name in selected))
        self.assertFalse(any('/python/transformers/' in name or '/site-packages/transformers/' in name for name in selected))

    def test_remote_e5_guard_uses_actual_worker_birth_and_ended_receipt(self):
        sys.path.insert(0, str(transport.SITE))
        import supervise
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); (base/'result-aa/job').mkdir(parents=True)
            (base/'deployment-aa.json').write_text(json.dumps(dict(pid=1, birth=100)))
            state = dict(complete=True, code=0, runs=[dict(job=dict(name='job'))])
            (base/'result-aa/identity.json').write_text(json.dumps(state))
            worker = dict(ended=123, members={'2': 101})
            (base/'result-aa/job/identity.json').write_text(json.dumps(worker))
            with patch.object(supervise, 'Path', return_value=base), patch.object(supervise, 'absent', return_value=True):
                self.assertEqual(supervise.e5_terminal()['aa']['code'], 0)
            with patch.object(supervise, 'Path', return_value=base), patch.object(supervise, 'absent', side_effect=lambda i: i['pid'] != 2):
                with self.assertRaisesRegex(AssertionError, 'Live e5 worker'): supervise.e5_terminal()

    def test_staging_and_launch_scripts_compile_and_never_reuse_existing_deployment(self):
        scripts = []
        def ssh(script, timeout=120):
            compile(script, 'transport-selftest', 'exec'); scripts.append(script)
            return '{}'
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with patch.object(transport, 'BASE', base), patch.object(transport, 'local_e5_terminal', return_value={}), \
                 patch.object(transport, 'checked_local', return_value=({'archive': {}}, {'archive': {}, 'execution': {}}, {})), \
                 patch.object(transport, 'ssh', side_effect=ssh), patch.object(transport.subprocess, 'run') as scp:
                transport.stage(); self.assertEqual(scp.call_count, 2)
                transport.launch(); self.assertEqual(len(scripts), 3)
                with self.assertRaises(AssertionError): transport.launch()
            self.assertLess(scripts[0].index('e5=e5_terminal()'), scripts[0].index('base.mkdir()'))
            self.assertIn("assert 'ended' in worker", scripts[0])

    def test_full_retained_graph_outputs_and_public_requests(self):
        # Only metadata is adapted to exercise the future Linux audit locally.
        # These remain retained Windows arrays, not an AMD execution claim.
        prepared = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload'
        original = ROOT/'artifacts/pyannote-lstm-output-lanes-20260921/outputs/1-candidate'
        self.assertTrue((original/'result.json').is_file())
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); folder = base/'output'; shutil.copytree(original, folder)
            (base/'manifests').mkdir()
            manifest_path = base/'manifests/portable-pyannote.json'
            shutil.copy2(prepared/'manifests/portable-pyannote.json', manifest_path)
            shutil.copy2(prepared/'graph-reference.json', base/'graph-reference.json')
            shutil.copytree(prepared/'graph-reference', base/'graph-reference')
            path = folder/'result.json'; result = read(path); result['runtime'] = '10.0.8'; result['manifest_sha256'] = pin(manifest_path)['sha256']
            path.write_text(json.dumps(result))
            report = qualify_outputs.pyannote(base, folder, 'portable')
            self.assertTrue(report['passed']); self.assertEqual(report['arrays'], 18); self.assertEqual(report['public_calls'], 16)
            # Rehashing a corrupted output does not bypass the native comparison.
            output = result['rows'][0]['output']; data_path = folder/output['file']
            values = np.fromfile(data_path, dtype='<f4'); values[0] += 1
            # All repeats receive the same corruption, so repeat checks cannot hide the native failure.
            for row in result['rows'][:3]:
                target = folder/row['output']['file']; values.tofile(target); row['output']['sha256'] = pin(target)['sha256']
            path.write_text(json.dumps(result))
            report = qualify_outputs.pyannote(base, folder, 'portable')
            self.assertFalse(report['passed'])

    def test_raw_integer_clocks_drive_summary_and_keep_slow_sample(self):
        from audit_results import timing_table
        manifest = dict(cases=[dict(name='case', samples=16000)])
        results = []
        for index in range(len(TIMING_ROLES)):
            records = [dict(name='case', phase='measured', start_ticks=10, end_ticks=10+v, frequency=10,
                            seconds=-1) for v in ([10, 10, 100] if index == 0 else [10, 10, 10])]
            results.append(dict(records=records))
        row = timing_table(results, manifest)[0]
        self.assertEqual(row['production']['maximum'], 10.)
        self.assertEqual(row['production']['seconds'], 2.5)
        self.assertEqual(row['ratios_to_ort']['production'], 2.5)

    def test_complete_retained_parakeet_audit_keeps_original_native_failures(self):
        path = ROOT/'tests/parakeet/transcribe/audit.py'
        spec = importlib.util.spec_from_file_location('retained_parakeet_audit', path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        report = module.audit(ROOT/'artifacts/parakeet-transcription-20260919/frozen/reference/manifest.json',
                              ROOT/'artifacts/pyannote-optimized-parakeet-20260921/candidate.json')
        self.assertTrue(report['audit_consistent']); self.assertTrue(report['application_passed'])
        self.assertFalse(report['numeric_gate_passed']); self.assertEqual(len(report['failures']), 3)
        self.assertEqual((report['arrays'], report['values']), (784, 3090494))
        self.assertEqual(report['maximum'], 0.0002321004867553711)


if __name__ == '__main__':
    unittest.main()
