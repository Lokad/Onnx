"""Full-count synthetic tick fixtures; no model inference or benchmark results."""
import copy
from pathlib import Path
import tempfile
import unittest

from test_contract import fixture
from contract import timing, write
from evidence import summarize
from report import render
from verify_report import verify_statistics, verify_markdown, decimal_interval
from fractions import Fraction as F


class ReportingTests(unittest.TestCase):
    def test_complete_tick_to_report_both_phases_and_damaged_summary(self):
        root = Path(__file__).resolve().parents[3]/'artifacts/e5-report-tools-20260921'
        root.mkdir(exist_ok=True)
        for phase in ('aa', 'compare'):
            jobs, values = fixture(phase)
            for value in values:
                value.update(conditioning=[dict(execute=1)]*128, load_ticks=1000, first=dict(execute=500))
                for row in value['measured']:
                    row.update(bytes=0, g0=0, g1=0, g2=0)
                # The source fixture repeats a timing dict; build independent
                # raw rows with the actual block/call indices for this layer.
                value['measured'] = [row | dict(block=i//value['specification']['calls'], call=i % value['specification']['calls'])
                                     for i, row in enumerate(value['measured'])]
            analyzed = dict(phase=phase, measured_calls=334080, conditioning_calls=230400,
                            timing=timing(values, jobs, phase), summaries=summarize(values, jobs),
                            diagnostic_screen=True, maximum_native_error=0,
                            frozen=dict(sha256='synthetic'), collection=dict(sha256='synthetic'))
            meta = dict(schedules={phase: jobs}, product_source='synthetic-only', core_sha256='synthetic')
            with tempfile.TemporaryDirectory(prefix='synthetic-', dir=root) as temporary:
                base = Path(temporary).resolve(); assert base.is_relative_to(root.resolve())
                write(base/'frozen.json', meta)
                for job, value in zip(jobs, values, strict=True):
                    folder = base/('result-'+phase)/job['name']/'output'; folder.mkdir(parents=True)
                    write(folder/'result.json', value)
                result = verify_statistics(base, phase, analyzed)
                self.assertEqual(result['measured_calls'], 334080)
                self.assertEqual(result['intervals'], 20 if phase == 'aa' else 60)
                self.assertTrue(result['statistical_screen'])
                report = render(analyzed, meta, dict(started=0, ended=1))
                self.assertEqual(verify_markdown(report, analyzed), 20)
                with self.assertRaises(AssertionError):
                    verify_markdown(report.replace('| e5-8tok | default |', '| wrong-case | default |', 1), analyzed)
                damaged = copy.deepcopy(analyzed)
                damaged['summaries'][0]['boundaries']['execute']['mean_ms'] += .01
                with self.assertRaises(AssertionError):
                    verify_statistics(base, phase, damaged)
                damaged = copy.deepcopy(analyzed)
                damaged['timing']['results'][1] = damaged['timing']['results'][0]
                with self.assertRaises(AssertionError):
                    verify_statistics(base, phase, damaged)

    def test_independent_decimal_unbounded_and_exact_point(self):
        self.assertFalse(decimal_interval([F(1), F(2), F(1)], [F(1), F(1), F(1000)], 4.)['bounded'])
        x = [F(100), F(102), F(98), F(101)]
        result = decimal_interval([v*F(97, 100) for v in x], x, 3.)
        self.assertEqual(result['interval'], [.97, .97])


if __name__ == '__main__':
    unittest.main()
