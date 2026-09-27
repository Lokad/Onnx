"""Admission must reject missed predictions, fallback regressions and drift."""
from fractions import Fraction as F
import unittest
from score import ORDER, evaluate, score


class Gates(unittest.TestCase):
    def setUp(self):
        self.cases = [dict(name=str(i), kind='target' if i == 0 else 'control') for i in range(6)]
        self.times = {n: [F(3, 4) if n.startswith('candidate') and i == 0 else F(1) for i in range(6)] for n in ORDER}
    def result(self): return evaluate(self.times, self.cases)
    def test_inclusive_boundaries(self):
        for n in ORDER[1:3]: self.times[n][1] = F(21, 20)
        self.assertTrue(self.result()['admitted'])
    def test_target_miss_is_failure(self):
        self.times[ORDER[1]][0] += F(1, 10**9)
        self.assertFalse(self.result()['admitted'])
    def test_fallback_is_not_hidden_by_target_gain(self):
        for n in ORDER[1:3]: self.times[n][0] = F(1, 10); self.times[n][5] = F(1050001, 1000000)
        self.assertFalse(self.result()['admitted'])
    def test_candidate_drift_is_failure(self):
        self.times[ORDER[1]][0] = F(1, 2)
        self.assertFalse(self.result()['admitted'])
    def test_current_drift_is_failure(self):
        self.times[ORDER[3]][0] = F(1100001, 1000000)
        self.assertFalse(self.result()['admitted'])
    def test_tail_clock_remains_in_mean(self):
        # A large observation contributes to the arithmetic mean, without trimming.
        for n in ORDER[1:3]: self.times[n][0] = (179*F(1, 2)+F(100))/180
        self.assertFalse(self.result()['admitted'])


class RawClocks(unittest.TestCase):
    def setUp(self):
        self.cases = [dict(index=i, name=str(i), kind='target' if i == 0 else 'control', shape=[1], batch=1) for i in range(6)]
        self.census = dict(cases=self.cases, protocol='decoder-packed-row-public-600-180-v1')
        self.reports = {}
        for seq, name in enumerate(ORDER):
            rows = []
            for i, case in enumerate(self.cases):
                ticks = 75 if name.startswith('candidate') and i == 0 else 100
                rows.append(dict(case, inputs=True, ownership=True, exact=True, setup_ticks=1,
                    input_sha256='a'*64, weight_sha256='b'*64, output_sha256='c'*64,
                    clocks=[dict(iteration=j, warmup=j < 600, start=1+(6*j+i)*200, ticks=ticks,
                        allocated=8, copies=0, scratch=0) for j in range(780)]))
            self.reports[name] = dict(passed=True, protocol=self.census['protocol'], role=name.split('-')[0], sequence=seq,
                samples=4680, warmup_samples=3600, measured_samples=1080, calls=4680, warmups=3600, measured=1080,
                frequency=1000000, rows=rows)
    def result(self): return score(self.reports, self.census)
    def test_complete_capture_passes(self): self.assertTrue(self.result()['admitted'])
    def test_warmup_copy_increase_fails(self):
        self.reports[ORDER[1]]['rows'][0]['clocks'][0]['copies'] = 1
        self.assertFalse(self.result()['admitted'])
    def test_missing_clock_rejected(self):
        self.reports[ORDER[1]]['rows'][0]['clocks'].pop()
        with self.assertRaises(AssertionError): self.result()
    def test_inexact_output_rejected(self):
        self.reports[ORDER[1]]['rows'][0]['output_sha256'] = 'd'*64
        with self.assertRaises(AssertionError): self.result()
    def test_last_clock_contributes_without_trim(self):
        self.reports[ORDER[1]]['rows'][0]['clocks'][-1]['ticks'] += 1
        self.assertFalse(self.result()['admitted'])


if __name__ == '__main__': unittest.main()
