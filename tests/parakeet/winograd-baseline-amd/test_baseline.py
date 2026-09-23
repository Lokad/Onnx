"""Exercise retained real records and independently known old clock summaries."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from protocol import pin, read
from checks import records_protocol, validate, evaluate, prereqs
from statistics_exact import timing_table
from prepare import APP, PRIOR, ROOT


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest_path = APP/'collected/manifests/candidate-parakeet.json'
        cls.manifest = read(cls.manifest_path)
        cls.original = records_protocol(APP/'collected')
        cls.public = read(PRIOR['parakeet'][0]/'collected/candidate-public/output/result.json')
        cls.spec = read(APP/'payload.json')
        cls.spec['identities'] = dict(current=cls.spec['identities']['candidate'])
        cls.old = ROOT/'artifacts/parakeet-single-panel-amd-execution-v2-20260922'
        cls.old_results = [read(cls.old/'collected/campaign'/name/'result.json') for name in
            ['timing-00-production-output', 'timing-02-ort-output', 'timing-03-ort-output', 'timing-05-production-output']]

    def managed(self):
        value = copy.deepcopy(self.public); value['conformance'] = False
        value['manifest_sha256'] = pin(self.manifest_path)['sha256']
        value['records'] = []; previous = 0
        # Synthetic clock placement only: complete outputs and original durations
        # come from the just-qualified current public requests; no benchmark claim.
        for iteration in range(4):
            for original in self.public['records']:
                row = copy.deepcopy(original); elapsed = row['end_ticks']-row['start_ticks']
                row.update(pass_=iteration)
                del row['pass_']; row['pass'] = iteration
                row['phase'] = 'warmup' if iteration == 0 else 'measured'
                row['start_ticks'] = previous+1; row['end_ticks'] = row['start_ticks']+elapsed
                previous = row['end_ticks']; value['records'].append(row)
        return value

    def check(self, value, native=False):
        return validate(value, self.manifest, self.spec, native, self.original, self.public, pin(self.manifest_path))

    def test_real_managed_outputs_and_invalid_records(self):
        value = self.managed(); self.assertTrue(self.check(value)['passed'])
        bad = copy.deepcopy(value); bad['records'].pop()
        with self.assertRaises(AssertionError): self.check(bad)
        bad = copy.deepcopy(value); bad['records'][1] = copy.deepcopy(bad['records'][0])
        with self.assertRaises(AssertionError): self.check(bad)
        for key, changed in [('held_outputs_unchanged', False), ('processor_count', 2), ('flags', {'DOTNET_EnableAVX512':'0'})]:
            bad = copy.deepcopy(value); bad[key] = changed
            with self.assertRaises(AssertionError): self.check(bad)
        for key, changed in [('ownership', False), ('seconds', .1), ('input_sha256', '0'*64), ('pass', 7)]:
            bad = copy.deepcopy(value); bad['records'][0][key] = changed
            with self.assertRaises(AssertionError): self.check(bad)
        bad = copy.deepcopy(value)
        text_key = next(k for k,v in bad['records'][0]['result'].items() if isinstance(v,str))
        bad['records'][0]['result'][text_key] += ' changed'
        with self.assertRaises(AssertionError): self.check(bad)

    def test_real_native_records_and_settings(self):
        value = copy.deepcopy(self.old_results[1])
        # Native identities/settings and outputs are unchanged; adapt only this
        # test fixture's descriptive managed-product manifest hash.
        value['manifest_sha256'] = pin(self.manifest_path)['sha256']
        self.assertTrue(self.check(value, True)['passed'])
        for key, changed in [('intra_threads',2), ('inter_threads',2), ('sequential',False), ('spinning',True), ('graph_optimizations','disabled')]:
            bad = copy.deepcopy(value); bad['native_settings'][key] = changed
            with self.assertRaises(AssertionError): self.check(bad, True)
        bad = copy.deepcopy(value); bad['numeric_libraries'] = {}
        with self.assertRaises(AssertionError): self.check(bad, True)
        bad = copy.deepcopy(value); bad['runner_sha256'] = '0'*64
        with self.assertRaises(AssertionError): self.check(bad, True)

    def test_old_raw_clocks_reconstruct_published_baseline(self):
        actual = timing_table(self.old_results, self.manifest)
        expected = read(self.old/'analysis.json')['table']
        for row, before in zip(actual, expected, strict=True):
            self.assertEqual(row['name'], before['name'])
            for role, old_role in [('current','production'), ('ort','ort')]:
                for key in ['seconds','exact_mean','rtf','minimum','maximum']:
                    self.assertEqual(row[role][key], before[old_role][key])
                for process, old_process in zip(row[role]['processes'], before[old_role]['processes'], strict=True):
                    for key in ['seconds','mean','exact_mean']: self.assertEqual(process[key], old_process[key])
        self.assertTrue(evaluate(actual)['baseline_valid'])
        self.assertFalse(evaluate(actual)['parity_target_met'])

    def test_every_control_required_and_every_case_present(self):
        table = timing_table(self.old_results, self.manifest)
        for index in range(21):
            for role in ['current','ort']:
                bad = copy.deepcopy(table); item = bad[index][role]
                first = Fraction(**item['processes'][0]['exact_mean']); changed = first*2
                item['processes'][1]['exact_mean'] = dict(numerator=changed.numerator, denominator=changed.denominator)
                mean = (first+changed)/2; item['exact_mean'] = dict(numerator=mean.numerator, denominator=mean.denominator)
                self.assertFalse(evaluate(bad)['baseline_valid'])
        with self.assertRaises(AssertionError): evaluate(table[:-1])
        bad = copy.deepcopy(table); bad[1] = copy.deepcopy(bad[0])
        with self.assertRaises(AssertionError): evaluate(bad)

    def test_all_retained_qualification_required(self):
        import shutil
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); spec = copy.deepcopy(self.spec); spec['prerequisites'] = {}
            for name, (folder, _) in PRIOR.items():
                target = base/'evidence'/name; target.mkdir(parents=True)
                for filename in ['closed.json','analysis.json']: shutil.copy2(folder/filename, target/filename)
                spec['prerequisites'][name] = dict(closed=pin(target/'closed.json'), analysis=pin(target/'analysis.json'))
            shutil.copy2(PRIOR['parakeet'][0]/'collected/candidate-public/output/result.json', base/'evidence/current-public.json')
            self.assertTrue(prereqs(base,spec)['passed'])
            for name in spec['prerequisites']:
                bad = copy.deepcopy(spec); del bad['prerequisites'][name]
                with self.assertRaises(AssertionError): prereqs(base,bad)
            target = base/'evidence/root/analysis.json'; value = read(target); value['root_source_verified'] = False
            target.write_text(json.dumps(value),encoding='utf8')
            with self.assertRaises(AssertionError): prereqs(base,spec)


if __name__ == '__main__': unittest.main()
