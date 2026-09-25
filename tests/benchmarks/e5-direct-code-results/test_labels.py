"""Keep the tier-name correction narrow and preserve both original labels."""
import importlib.util
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/benchmarks/e5-direct-code-diagnostic-amd'))
from test_checks import Checks
from codegen_labels import reconcile_listings


class Labels(unittest.TestCase):
    def fixture(self):
        text,events=Checks().code_fixture()
        start=text.index('; Assembly listing for method ',10)
        part=text[start:].replace('(Tier1)','(Instrumented Tier0)')
        part=part.replace('       C3', '       E800000000           call     CORINFO_HELP_COUNTPROFILE32\n       E800000000           call     CORINFO_HELP_PATCHPOINT\n       C3')
        part=part.replace('Total bytes of code 1','Total bytes of code 11')
        text=text[:start]+part
        events[1]['payload']['OptimizationTier']='QuickJitted';events[1]['payload']['MethodSize']='11'
        return text,events
    def test_preserve_distinct_labels(self):
        text,events=self.fixture();result=reconcile_listings(text,events,1)
        row=result['rows'][1]
        self.assertEqual(row['tier'],'Instrumented Tier0');self.assertEqual(row['runtime_tier'],'QuickJitted')
        self.assertTrue(row['rendered_runtime_label_difference'])
        self.assertIn('(Instrumented Tier0)',row['listing'])
    def test_counter_evidence_required(self):
        text,events=self.fixture()
        with self.assertRaises(AssertionError):reconcile_listings(text.replace('CORINFO_HELP_COUNTPROFILE32','Other'),events,1)
    def test_patchpoint_evidence_required(self):
        text,events=self.fixture()
        with self.assertRaises(AssertionError):reconcile_listings(text.replace('CORINFO_HELP_PATCHPOINT','Other'),events,1)
    def test_size_still_exact(self):
        text,events=self.fixture();events[1]['payload']['MethodSize']='12'
        with self.assertRaises(AssertionError):reconcile_listings(text,events,1)


if __name__=='__main__':unittest.main()
