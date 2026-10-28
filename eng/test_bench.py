import unittest
from pathlib import Path
import subprocess
import sys
import tempfile

import bench
import campaign_evidence as evidence
import isolated_evidence as isolated


class BenchTests(unittest.TestCase):
    def test_runtime_identity_accepts_release_and_complete_preview_versions(self):
        env = dict(host='host', cpu='AMD', os='Linux', architecture='x64', sdk='11.0.100-rc.3.26522.148',
                   runtime='', isa='AVX2,FMA', affinity='0x4',
                   settings=dict(jit='default-tiered', gc='workstation', variables={'DOTNET_TieredCompilation': '1'}))
        for runtime in ('.NET 10.0.8', '.NET 11.0.0', '.NET 11.0.0-rc.3.26522.148', '.NET 11.0.0-preview.7.26351.2'):
            with self.subTest(runtime=runtime):
                env['runtime'] = runtime
                self.assertEqual(runtime, isolated.environment(dict(environment=env))['runtime'])
        for runtime in ('.NET 11', '.NET 11.0', '.NET 11.0.0-rc.', '.NET 11.0.0-rc.3 extra', '.NET 11.0.0-unknown'):
            with self.subTest(runtime=runtime):
                env['runtime'] = runtime
                with self.assertRaisesRegex(evidence.EvidenceError, 'runtime version'):
                    isolated.environment(dict(environment=env))

    def test_routine_import_does_not_load_campaign_frontends_or_scorer(self):
        code = "import sys; import bench; assert not {'run_common_campaign', 'run_isolated_e5', 'run_embeddinggemma2', 'score_campaign'}.intersection(sys.modules)"
        subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parent, check=True)

    def test_historical_producer_inventory_is_readable_but_cannot_hide_new_imports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bindings = {}
            for name in isolated.LEGACY_SCRIPTS:
                path = root / name
                path.write_text('# historical producer\n', encoding='utf-8')
                bindings[name] = isolated.binding(path, root)
            self.assertEqual(set(bindings), set(isolated.producer_files(root, bindings, isolated.SCRIPTS, isolated.LEGACY_SCRIPTS)))
            path = root / 'run_isolated_e5.py'
            for source in ('import e5_worker\n', 'from benchmark_worker import stage\n'):
                path.write_text(source, encoding='utf-8')
                bindings[path.name] = isolated.binding(path, root)
                with self.assertRaisesRegex(evidence.EvidenceError, 'shared worker source provenance missing'):
                    isolated.producer_files(root, bindings, isolated.SCRIPTS, isolated.LEGACY_SCRIPTS)

    def test_visits_balance_order(self):
        self.assertEqual([('L0', 'L1'), ('L1', 'L0')], bench.visits(2))
        self.assertEqual(2, bench.visits(4).count(('L0', 'L1')))
        with self.assertRaises(Exception):
            bench.visits(1)

    def workers(self):
        return [dict(case='case', pair=pair, role=role, engine=engine,
                     samples_ms=[value, value * 100, value])
                for pair in (1, 2) for role in ('L0', 'L1') for engine in ('lok', 'ort')
                for value in [20 if engine == 'ort' else (10 if role == 'L0' else 8)]]

    def test_report_retains_outliers_and_compares_process_medians(self):
        workers = self.workers()
        result = bench.summarize(workers)['case']
        self.assertEqual(0.8, result['candidate_control'])
        self.assertEqual([0.8, 0.8], result['paired_candidate_control'])
        self.assertEqual(0.4, result['arms']['L1']['lok_ort'])
        self.assertEqual(24, sum(len(worker['samples_ms']) for worker in workers))
        self.assertEqual([10, 1000, 10], workers[0]['samples_ms'])

    def test_missing_failed_or_duplicate_worker_cannot_produce_summary(self):
        workers = self.workers()
        with self.assertRaises(KeyError):
            bench.summarize(workers[:-1])
        workers[0]['samples_ms'] = []
        with self.assertRaises(Exception):
            bench.summarize(workers)
        with self.assertRaises(Exception):
            bench.summarize(self.workers() * 2)

    def test_native_identity_binds_actual_loaded_library_between_workers(self):
        signatures = {}
        bench.check_native(dict(mode='lok', native_module=None), signatures)
        bench.check_native(dict(mode='ort', native_module=dict(sha256='first')), signatures)
        bench.check_native(dict(mode='ort', native_module=dict(sha256='first')), signatures)
        with self.assertRaises(Exception):
            bench.check_native(dict(mode='ort', native_module=dict(sha256='changed')), signatures)


if __name__ == '__main__':
    unittest.main()
