import unittest

import bench


class BenchTests(unittest.TestCase):
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
