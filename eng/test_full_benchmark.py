"""Offline full-suite scheduling, deadline and evidence checks; no large models."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import gemma_worker as gemma
import full_benchmark as full
import campaign_evidence as evidence
from types import SimpleNamespace


def completed_workers():
    result=[]
    for job in full.jobs():
        if job['phase']=='conformance':samples=[1.,1.] if job['case']=='gemma2-128tok' else []
        else:
            count=3 if job['case'] in ('pyannote','parakeet') else 33 if job['case']=='gemma2-128tok' else 180
            samples=[1.02 if job['engine']=='lok' else 1.]*count
        result.append(dict(job,samples_ms=samples))
    return result


class FullScheduleTests(unittest.TestCase):
    def test_all_seven_full_workloads_and_fixed_balanced_pairs(self):
        jobs=full.jobs()
        self.assertEqual(58,len(jobs))
        self.assertEqual(set(full.CASES),{j['case'] for j in jobs})
        self.assertEqual(list(range(1,59)),[j['index'] for j in jobs])
        self.assertTrue(all(j['phase']=='conformance' for j in jobs[:14]))
        self.assertTrue(all(j['phase']=='timing' for j in jobs[14:]))
        for case in full.CASES:
            count=6 if case in ('pyannote','parakeet') else 2
            timed=[j for j in jobs if j['case']==case and j['phase']=='timing']
            self.assertEqual(2*count,len(timed))
            for pair in range(1,count+1):
                self.assertEqual(['lok','ort'] if pair%2 else ['ort','lok'],[j['engine'] for j in timed if j['pair']==pair])
            self.assertFalse(any(j.get('role')=='L0' for j in timed))

    def test_complete_result_uses_six_independent_pair_intervals(self):
        summary=full.summarize(completed_workers())
        self.assertEqual(set(full.CASES),set(summary))
        for case in ('parakeet','pyannote'):
            self.assertEqual(5,summary[case]['df'])
            self.assertEqual(6,len(summary[case]['paired_ratios']))
            self.assertAlmostEqual(1.02,summary[case]['ratio'])
            self.assertTrue(summary[case]['target_pass'])

    def test_missing_duplicate_reordered_or_truncated_results_cannot_pass(self):
        values=completed_workers()
        variants=[values[:-1],values+[values[-1]],values[:14]+list(reversed(values[14:]))]
        cut=completed_workers();cut[-1]['samples_ms'].pop();variants.append(cut)
        for values in variants:
            with self.subTest(length=len(values)),self.assertRaises(ValueError):full.summarize(values)

    def test_all_samples_including_large_observations_are_retained(self):
        values=completed_workers()
        item=next(v for v in values if v['case']=='parakeet' and v['phase']=='timing' and v['engine']=='lok')
        item['samples_ms']=[1.,100.,1.]
        summary=full.summarize(values)
        self.assertEqual([1.,100.,1.],item['samples_ms'])
        self.assertAlmostEqual(34.,summary['parakeet']['lok_process_ms'][0])
        self.assertFalse(summary['parakeet']['target_pass'])

    def test_wall_clock_budget_does_not_reset_between_models(self):
        self.assertEqual(30,full.remaining(100,clock=lambda:70))
        with self.assertRaisesRegex(ValueError,'duration limit exceeded'):full.remaining(100,clock=lambda:100)

    def exercise_run(self,seconds):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);prepared=root/'prepared';prepared.mkdir()
            fixtures=root/'assets';fixtures.mkdir();(fixtures/'fixtures.json').write_text('{}')
            runner=prepared/'runner';runner.mkdir()
            names=['Lokad.Onnx.AudioBench.dll','Lokad.Onnx.GraphBench.dll','Lokad.Onnx.EmbeddingGemma2Bench.dll']
            for name in names:(runner/name).write_bytes(b'caller')
            (prepared/'runner-source.zip').write_bytes(b'caller sources')
            receipt=dict(sdk='qualified',target_framework='net11.0',source_archive_sha256=evidence.sha256(prepared/'runner-source.zip'),
                         files={name:evidence.sha256(runner/name) for name in names})
            (prepared/'full-build.json').write_text(json.dumps(receipt))
            build=dict(source_sha='a'*40,core_sha256='b'*64,sdk='qualified',target_framework='net11.0',product_files={},runner=runner/names[0])
            output=root/'output'
            args=SimpleNamespace(cpu=2,fixtures=fixtures/'fixtures.json',case=None,verify_only=False,dimensions=768,
                                 output=output,prepared=prepared)
            now=[0.]
            def load(*unused):now[0]+=15;return {},{'gemma':fixtures/'fixtures.json'},{},{},{}
            def stage(*unused,**kwargs):output.mkdir();return {'L1':build}
            def execute(*unused):
                common_value=completed_workers()[0]
                full.common.save_json(output/'completed-001.json',common_value)
                now[0]+=seconds
                if seconds>3600:raise TimeoutError('full benchmark duration limit exceeded')
                return completed_workers()
            original_save=full.common.save_json
            def save(path,value):
                if path.name=='report.json':now[0]+=5
                return original_save(path,value)
            with mock.patch.object(full,'require_host'),mock.patch.object(full.time,'monotonic',side_effect=lambda:now[0]), \
                    mock.patch.object(full.common,'read_build',return_value=build),mock.patch.object(full,'load_fixtures',side_effect=load), \
                    mock.patch.object(full.common,'stage',side_effect=stage),mock.patch.object(full,'execute',side_effect=execute), \
                    mock.patch.object(full.common,'save_json',side_effect=save), \
                    mock.patch.object(full.os,'sched_getaffinity',return_value={0,1,2,3},create=True), \
                    mock.patch.object(full.os,'sched_setaffinity',create=True):
                with mock.patch('builtins.print'):
                    if seconds>3600:
                        with self.assertRaisesRegex(TimeoutError,'duration limit exceeded'):full.run(args)
                    else:full.run(args)
            if seconds>3600:
                self.assertTrue((output/'completed-001.json').exists())
                self.assertFalse((output/'complete.json').exists())
                failure=json.loads((output/'failure.json').read_text())
                self.assertFalse(failure['complete'])
                with self.assertRaisesRegex(ValueError,'failure receipt'):full.completed(output)
                return
            completion=json.loads((output/'complete.json').read_text())
            self.assertEqual(1220,completion['elapsed_seconds'])
            self.assertTrue(completion['complete'])
            self.assertEqual(evidence.sha256(output/'report.json'),completion['report_sha256'])
            full.completed(output)
            (output/'failure.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'failure receipt'):full.completed(output)

    def test_total_duration_includes_setup_and_report_write(self):
        self.exercise_run(1200)

    def test_duration_expiry_keeps_partial_evidence_without_success(self):
        self.exercise_run(3700)


class GemmaDeadlineTests(unittest.TestCase):
    def test_expired_suite_never_creates_or_launches_gemma_worker(self):
        with mock.patch.object(gemma.time, 'monotonic', return_value=101), mock.patch.object(gemma.common, 'spawn_child') as spawn:
            with self.assertRaisesRegex(ValueError, 'duration limit exceeded before launch'):
                gemma.child({}, Path('.'), Path('.'), 'gemma2-128tok', 'lok', 2, 768, 0, False, {}, 1,
                            overall_deadline=100)
            spawn.assert_not_called()

    def test_remaining_suite_time_bounds_worker_and_failure_receipt_is_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'fixtures').mkdir()
            (root/'fixtures/fixtures.json').write_text('{}')
            process = mock.Mock(pid=4242, returncode=-15)
            with mock.patch.object(gemma.time, 'monotonic', side_effect=[0, 95]), \
                    mock.patch.object(gemma.processes, 'snapshot', return_value={'fixture': True}), \
                    mock.patch.object(gemma.processes, 'foreign_fraction', return_value={'fixture': True}), \
                    mock.patch.object(gemma.common, 'spawn_child', return_value=process), \
                    mock.patch.object(gemma.common, 'wait_child', side_effect=TimeoutError('suite deadline')) as wait:
                with self.assertRaisesRegex(TimeoutError, 'suite deadline'):
                    gemma.child({'runner': root/'caller.dll', 'source_sha': 'a'*40, 'core_sha256': 'b'*64},
                                root, root, 'gemma2-128tok', 'lok', 2, 768, 0, False, {}, 1,
                                overall_deadline=100, dotnet='/qualified/dotnet')
            self.assertEqual(5, wait.call_args.kwargs['timeout'])
            receipt = json.loads((root/'worker-001-gemma2-128tok-lok.supervision.json').read_text())
            self.assertEqual(4242, receipt['pid'])
            self.assertEqual(-15, receipt['exit_code'])
            self.assertEqual('suite deadline', receipt['failure'])
            self.assertEqual('/qualified/dotnet', receipt['command'][0])
            self.assertTrue((root/'worker-001-gemma2-128tok-lok.pre.json').exists())
            self.assertTrue((root/'worker-001-gemma2-128tok-lok.post.json').exists())


if __name__ == '__main__':
    unittest.main()
