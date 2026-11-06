"""Offline qualification-identity and process-pair checks for the shared-worker coordinator."""
import importlib.util,json,math,tempfile,unittest
from pathlib import Path
from unittest import mock

spec=importlib.util.spec_from_file_location('paradee_benchmark',Path(__file__).with_name('bench-paradee.py'))
paradee=importlib.util.module_from_spec(spec);spec.loader.exec_module(paradee)


class ParadeeBenchmarkTests(unittest.TestCase):
    def test_six_independent_pair_scores_and_interval(self):
        workers=[dict(case='model',phase='timing',pair=pair,engine=engine,samples_ms=[1.02 if engine=='lok' else 1.]*33)
                 for pair in range(1,7) for engine in ('lok','ort')]
        result=paradee.summarize(workers,['model'],6)['model']
        self.assertAlmostEqual(1.02,result['ratio'])
        self.assertAlmostEqual(1.02,result['interval_95'][0]);self.assertAlmostEqual(1.02,result['interval_95'][1])
        self.assertTrue(result['target_pass'])
        with self.assertRaises(ValueError):paradee.summarize(workers[:-1],['model'],6)
        workers[-1]=workers[0]
        with self.assertRaises(ValueError):paradee.summarize(workers,['model'],6)

    def test_pilot_does_not_claim_a_confidence_interval(self):
        workers=[dict(case='model',phase='timing',pair=1,engine=engine,samples_ms=[2. if engine=='lok' else 1.]) for engine in ('lok','ort')]
        result=paradee.summarize(workers,['model'],1)['model']
        self.assertEqual(2.,result['ratio']);self.assertNotIn('interval_95',result);self.assertNotIn('target_pass',result)

    def test_qualification_requires_all_six_passes_and_exact_core_identity(self):
        core='a'*64
        names=['Fp32ControlledNoiseMatchesOrt','Fp32OriginalExportRuns','Int8ControlledNoiseMatchesOrt','Int8OriginalExportRuns',
               'Fp32WaveformOnlyMatchesOrt','Int8WaveformOnlyMatchesOrt']
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'qualification.trx'
            def write(outcome='Passed',digest=core,count=6):
                path.write_text('<TestRun xmlns="urn:test"><Results>'+''.join(
                    '<UnitTestResult testName="'+name+'" outcome="'+outcome+'"><Output><StdOut>core_sha256='+digest+
                    '; runtime=11.0.0</StdOut></Output></UnitTestResult>' for name in names[:count])+'</Results></TestRun>')
            write();self.assertTrue(paradee.qualify(path,core)['passed'])
            for arguments in [dict(outcome='NotExecuted'),dict(digest='b'*64),dict(count=5),dict(count=4)]:
                write(**arguments)
                with self.subTest(arguments=arguments),self.assertRaises(ValueError):paradee.qualify(path,core)
            names[-1]=names[-2];write()
            with self.assertRaises(ValueError):paradee.qualify(path,core)

    def test_underconditioned_manifest_never_launches_a_worker(self):
        class WorkerReached(Exception):pass
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);runner=root/'runner.dll';runner.write_bytes(b'caller')
            core_file=root/'Lokad.Onnx.dll';core_file.write_bytes(b'core');core=paradee.digest(core_file)
            model=root/'model.onnx';model.write_bytes(b'model')
            qualification=root/'qualification.trx';qualification.write_text('<TestRun/>')
            manifest=root/'manifest.json'
            def check_output(command,**kwargs):
                if command[0]=='git':return '' if '--porcelain' in command else 'a'*40
                return '{"numpy":"2.2.4","onnxruntime":"1.29.0"}'
            for seconds in (10,40):
                manifest.write_text(json.dumps(dict(core=core,cases=[dict(key='paradee-fp32-14tok',model=str(model),
                    model_sha256=paradee.digest(model),stochastic_outputs=True,
                    timing=dict(warmup_calls=32,warmup_seconds=seconds,measured_calls=33))])))
                arguments=['bench-paradee.py','--manifest',str(manifest),'--runner',str(runner),'--qualification',str(qualification),
                    '--dotnet','dotnet','--native-python','python','--output',str(root/f'results-{seconds}')]
                with mock.patch('sys.argv',arguments),mock.patch.object(paradee,'qualify',return_value={}),\
                     mock.patch.object(paradee.subprocess,'check_output',side_effect=check_output),\
                     mock.patch.object(paradee.graph,'child',side_effect=WorkerReached) as child:
                    if seconds==10:
                        with self.assertRaisesRegex(ValueError,'timing schedule'):paradee.main()
                        child.assert_not_called()
                    else:
                        with self.assertRaises(WorkerReached):paradee.main()
                        child.assert_called_once()


if __name__=='__main__':unittest.main()
