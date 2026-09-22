"""Gate boundary and weighting tests; no benchmark execution."""
from fractions import Fraction as F
import unittest
from score import ORDER, evaluate
from score import iteration_manifest, call_totals
import copy
import json
from pathlib import Path


class Gates(unittest.TestCase):
    def totals(self, production, candidate):
        return {name: dict(candidate if name.startswith('candidate') else production) for name in ORDER}

    def test_exact_weighted_boundary(self):
        r = evaluate(self.totals({0:F(9),1:F(1)}, {0:F(8),1:F(1)}), {0:True,1:False})
        self.assertTrue(r['admitted'])
        self.assertEqual(r['rows'][0]['ratio']['numerator'], 9)
        self.assertEqual(r['rows'][0]['ratio']['denominator'], 10)

    def test_one_tick_above_boundary_fails(self):
        r = evaluate(self.totals({0:F(10**12)}, {0:F(9*10**11+1)}), {0:True})
        self.assertFalse(r['admitted'])

    def test_fallback_cost_cannot_be_omitted(self):
        r = evaluate(self.totals({0:F(1),1:F(9)}, {0:F(1,2),1:F(9)}), {0:True,1:False})
        self.assertFalse(r['admitted'])

    def test_eligible_form_regression_is_independent(self):
        r = evaluate(self.totals({0:F(100),1:F(1)}, {0:F(80),1:F(106,100)}), {0:True,1:True})
        self.assertTrue(r['gates'][0]['passed']); self.assertFalse(r['admitted'])

    def test_control_failure_rejects_fast_candidate(self):
        t = self.totals({0:F(100)}, {0:F(50)}); t['production-b'][0] = F(111)
        r = evaluate(t, {0:True}); self.assertFalse(r['admitted'])
        self.assertTrue(all(g['passed'] for g in r['gates']))

    def test_form_control_cannot_hide_in_aggregate(self):
        t = self.totals({0:F(100),1:F(1)}, {0:F(80),1:F(1,2)}); t['production-b'][1] = F(121,100)
        r = evaluate(t, {0:True,1:True}); self.assertFalse(r['admitted'])
        self.assertTrue(r['controls'][0]['passed']); self.assertFalse(r['controls'][2]['passed'])

    def test_nonpositive_and_incomplete_totals_rejected(self):
        with self.assertRaises(AssertionError): evaluate(self.totals({0:F(0)}, {0:F(1)}), {0:True})
        with self.assertRaises(AssertionError): evaluate(self.totals({0:F(1)}, {0:F(1)}), {0:True,1:False})


class Iterations(unittest.TestCase):
    def fixtures(self):
        calls = []
        for i, channels in enumerate([16, 32]):
            calls.append(dict(case='crop', index=i, form=i, eligible=True,
                input=dict(shape=[1, channels, 128, 128]), output=dict(shape=[1, 32, 128, 128]),
                weights=dict(shape=[32, channels, 3, 3]), attributes=dict(group=1)))
        return calls

    def data(self):
        calls = self.fixtures(); manifest = iteration_manifest(calls); rows = []
        for p in range(4):
            for c, e in zip(calls, manifest['calls']):
                for i in range(e['iterations']):
                    rows.append(dict(kind='call', role='production', pass_=p, warmup=p == 0,
                        name=c['case'], index=c['index'], form=c['form'], eligible=True, iteration=i,
                        iterations=e['iterations'], work=e['work'], ticks=100 if c['index'] == 0 else 200,
                        frequency=100, exact=True, sha256='expected', values=524288))
                    rows[-1]['pass'] = rows[-1].pop('pass_')
        result = dict(protocol=manifest['protocol'], cases=2, warmups=manifest['per_pass'],
            measured=3*manifest['per_pass'], calls=len(rows), observations=rows)
        reference = {(c['case'], c['index']): dict(production='expected', values=524288) for c in calls}
        return result, calls, reference

    def test_geometry_ceiling_and_bounds(self):
        m = iteration_manifest(self.fixtures()); self.assertEqual([r['iterations'] for r in m['calls']], [29, 15])
        calls = self.fixtures(); calls[0]['output']['shape'][2:] = [1, 1]
        with self.assertRaises(AssertionError): iteration_manifest(calls)

    def test_unequal_iterations_keep_original_call_weight(self):
        data = self.data(); self.assertEqual(call_totals(*data, 'production'), {0:F(1), 1:F(2)})
        data[0]['observations'][0]['ticks'] = 10**12  # Warmup is retained but excluded.
        self.assertEqual(call_totals(*data, 'production'), {0:F(1), 1:F(2)})

    def test_missing_duplicate_and_wrong_iterations_rejected(self):
        data = self.data()
        for mutation in ['missing', 'duplicate', 'count', 'work']:
            r, calls, ref = copy.deepcopy(data)
            if mutation == 'missing': r['observations'].pop()
            elif mutation == 'duplicate': r['observations'][1] = r['observations'][0]
            elif mutation == 'count': r['observations'][0]['iterations'] += 1
            else: r['observations'][0]['work'] += 1
            with self.assertRaises(AssertionError): call_totals(r, calls, ref, 'production')

    def test_one_tick_has_exact_iteration_denominator(self):
        r, calls, ref = self.data(); i = r['warmups']; r['observations'][i]['ticks'] += 1
        result = call_totals(r, calls, ref, 'production')
        self.assertEqual(result[0], F(1)+F(1, 100*3*29))

    def test_actual_fixture_manifest(self):
        path = Path(__file__).resolve().parents[3]/'artifacts/pyannote-blocked-spatial-fixtures-20260922/output/result.json'
        calls = json.loads(path.read_text())['calls']; m = iteration_manifest(calls)
        self.assertEqual(len(m['calls']), 108)
        self.assertEqual(m['per_pass'], 1074)
        self.assertEqual(sorted({r['iterations'] for r in m['calls']}), [3, 6, 53, 94])


if __name__ == '__main__': unittest.main()
