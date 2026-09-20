from pathlib import Path
import tempfile
import unittest
from close_report import report
from common import read


class ReportingTests(unittest.TestCase):
    def test_complete_closed_amd_report_renders_scores_and_all_transcripts(self):
        base=Path(__file__).resolve().parents[3]/'artifacts/asr-multilingual-amd-20260920/collected'
        if not (base/'closed.json').exists():self.skipTest('Optional closed AMD evidence absent')
        with tempfile.TemporaryDirectory() as folder:
            output=Path(folder);assert output.resolve().parent==Path(tempfile.gettempdir()).resolve()
            report(base,output)
            text=(output/'results-20260920.md').read_text(encoding='utf-8')
            transcripts=(output/'transcripts-20260920.md').read_text(encoding='utf-8')
            value=read(output/'observations-20260920.json')
            self.assertIn('fixed AMD comparison',text);self.assertIn('AMD EPYC 9V74',text)
            self.assertIn('14 GiB RSS',text);self.assertIn('13 GiB available',text)
            self.assertIn('Windows memory-failure receipt',text)
            self.assertEqual(value['models'],read(base/'audit.json')['models'])
            self.assertEqual(transcripts.count('### '),80)
            for model in value['models'].values():
                for case in model['cases']:self.assertIn('### '+case['name'],transcripts)
            with self.assertRaises(AssertionError):report(base,output)


if __name__=='__main__':unittest.main()
