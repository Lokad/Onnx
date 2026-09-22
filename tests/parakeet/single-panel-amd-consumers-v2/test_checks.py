import copy
import json
import unittest
from adapt import ROOT, CORE, CONTROL_CORE, probe_source, caller_source
from checks import probe, caller


class CoverageTests(unittest.TestCase):
    def fixture(self):
        value = json.loads((ROOT / 'artifacts/parakeet-single-panel-composition-20260922/geometry.json').read_text())
        value.update(original_core_sha256=CONTROL_CORE, avx512=True, runtime='10.0.8')
        value['prepared_precedence'] = [dict(m=m, n=n, k=k, baseline='0'*64, candidate='0'*64, partial='1'*64,
            input_a='2'*64, input_b='3'*64, packed='4'*64, initial='5'*64,
            differs=True, guards_preserved=True, operands_preserved=True)
            for m in [8, 9, 12, 13, 14, 16, 17, 20, 24, 25] for n in [1024, 1025] for k in [32, 64]]
        return value

    def test_full_amd_schedule(self):
        self.assertEqual(probe(self.fixture(), 'normal', '10.0.8', True)['prepared_precedence_cases'], 40)

    def test_incomplete_duplicate_changed_unguarded_or_vacuous_refused(self):
        changes = [lambda r: r['prepared_precedence'].pop(),
            lambda r: r['prepared_precedence'].append(r['prepared_precedence'][0]),
            lambda r: r['prepared_precedence'][0].update(candidate='f'*64),
            lambda r: r['prepared_precedence'][0].update(guards_preserved=False),
            lambda r: [v.update(partial='0'*64, differs=False) for v in r['prepared_precedence']],
            lambda r: r.update(dynamic_dispatch_cases=47), lambda r: r.update(original_core_sha256='f'*64)]
        for mutate in changes:
            with self.subTest(mutation=changes.index(mutate)):
                value = self.fixture(); mutate(value)
                with self.assertRaises(AssertionError): probe(value, 'normal', '10.0.8', True)

    def test_local_has_no_amd_claim(self):
        value = self.fixture(); value.update(avx512=False, runtime='10.0.12', prepared_precedence=[])
        self.assertEqual(probe(value, 'normal', '10.0.12', False)['prepared_precedence_cases'], 0)
        with self.assertRaises(AssertionError): probe(value, 'normal', '10.0.12', True)

    def test_caller_schedule_preserved(self):
        value = json.loads((ROOT / 'artifacts/pyannote-single-panel-composition-20260922/caller-normal.json').read_text())
        value.update(candidate=CORE, baseline=CONTROL_CORE)
        shapes = json.loads((ROOT / 'artifacts/pyannote-single-panel-direct-20260922/shapes.json').read_text())
        self.assertEqual(caller(value, shapes, 'normal', '10.0.12')['cases'], 400)
        for mutate in [lambda r: r['records'].pop(), lambda r: r['records'][0].update(digest='f'*64), lambda r: r.update(baseline='f'*64)]:
            altered = copy.deepcopy(value); mutate(altered)
            with self.assertRaises(AssertionError): caller(altered, shapes, 'normal', '10.0.12')

    def test_exact_source_adaptations(self):
        self.assertIn('new Random(652199)', probe_source())
        self.assertIn('new Random(119837)', probe_source())
        self.assertEqual(caller_source().count(CONTROL_CORE), 1)


if __name__ == '__main__': unittest.main()
