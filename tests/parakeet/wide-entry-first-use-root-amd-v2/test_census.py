"""Exercise the fixed hardware census against actual TRX and adversarial outcomes."""
import importlib.util,tempfile,unittest
from pathlib import Path
import xml.etree.ElementTree as ET
from checks import suite256,census
from consumer_scope import verify_scope
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'artifacts/parakeet-wide-entry-first-use-root-amd-20260923/collected'
ACTUAL=OLD/'backend-tests-256/backend.trx'
EXP='Lokad.Onnx.Backend.Tests.Exp512Tests.ProbeHandlesExceptionals'
UNSUPPORTED='Lokad.Onnx.Backend.Tests.PackedAvx512RowTests.UnsupportedHardwareDeclinesWithoutTouchingMemory'


class CensusTests(unittest.TestCase):
    def changed(self,select,outcome=None,rename=False):
        doc=ET.parse(ACTUAL);rows=doc.findall('.//{*}UnitTestResult')
        row=next(r for r in rows if select(r.attrib))
        if outcome is not None:row.attrib['outcome']=outcome
        if rename:row.attrib['testName']+=' unexpected'
        counters=doc.find('.//{*}Counters').attrib
        counters['passed']=str(sum(r.attrib['outcome']=='Passed' for r in rows))
        counters['failed']=str(sum(r.attrib['outcome']=='Failed' for r in rows))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'changed.trx';doc.write(path)
            with self.assertRaises(AssertionError):suite256(path,'backend',OLD/'evidence')

    def test_actual_complete_census(self):
        value=suite256(ACTUAL,'backend',OLD/'evidence')
        self.assertEqual((value['passed'],value['skipped']),(3359,131))

    def test_retained_scope(self):self.assertTrue(verify_scope())

    def test_previous_checker_rejects_exact_actual_result(self):
        spec=importlib.util.spec_from_file_location('old_root_census',ROOT/'tests/parakeet/wide-entry-first-use-root-amd/checks.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        with self.assertRaises(AssertionError):old.suite256(ACTUAL,'backend',OLD/'evidence')

    def test_unrelated_skip_rejected(self):
        before=census(OLD/'evidence/selected-backend.trx')
        self.changed(lambda a:a['outcome']=='Passed' and before[(a['testName'],'Passed')]==1,'NotExecuted')

    def test_expected_exp_skip_required(self):self.changed(lambda a:a['testName']==EXP,'Passed')
    def test_unsupported_case_must_execute(self):self.changed(lambda a:a['testName']==UNSUPPORTED,'NotExecuted')
    def test_any_test_failure_rejected(self):self.changed(lambda a:a['outcome']=='Passed','Failed')
    def test_renamed_case_rejected(self):self.changed(lambda a:a['testName']==EXP,rename=True)


if __name__=='__main__':unittest.main()
