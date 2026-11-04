"""Offline qualification-identity and process-pair checks for the shared-worker coordinator."""
import importlib.util,json,math,tempfile,unittest
from pathlib import Path

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

    def test_qualification_requires_all_four_passes_and_exact_core_identity(self):
        core='a'*64
        names=['Fp32ControlledNoiseMatchesOrt','Fp32OriginalExportRuns','Int8ControlledNoiseMatchesOrt','Int8OriginalExportRuns']
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'qualification.trx'
            def write(outcome='Passed',digest=core,count=4):
                path.write_text('<TestRun xmlns="urn:test"><Results>'+''.join(
                    '<UnitTestResult testName="'+name+'" outcome="'+outcome+'"><Output><StdOut>core_sha256='+digest+
                    '; runtime=11.0.0</StdOut></Output></UnitTestResult>' for name in names[:count])+'</Results></TestRun>')
            write();self.assertTrue(paradee.qualify(path,core)['passed'])
            for arguments in [dict(outcome='NotExecuted'),dict(digest='b'*64),dict(count=3)]:
                write(**arguments)
                with self.subTest(arguments=arguments),self.assertRaises(ValueError):paradee.qualify(path,core)


if __name__=='__main__':unittest.main()
