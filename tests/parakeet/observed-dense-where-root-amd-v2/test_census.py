"""Reject missing, renamed, skipped or duplicate added cases in either mode."""
import copy
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from checks import suite,suite256
from consumer_scope import verify_scope
from new_cases import NEW_CASES

ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'artifacts/parakeet-validated-composition-root-amd-20260924/collected'


class CensusTests(unittest.TestCase):
    def fixture(self,name,disabled):
        folder=name+'-tests'+('-256' if disabled else '')
        doc=ET.parse(OLD/folder/(name+'.trx'))
        results=doc.find('.//{*}Results')
        template=doc.find('.//{*}UnitTestResult')
        for case in NEW_CASES[name]:
            row=copy.deepcopy(template)
            row.attrib.update(testName=case,outcome='Passed')
            results.append(row)
        return doc

    def check(self,doc,name,disabled):
        rows=doc.findall('.//{*}UnitTestResult')
        counters=doc.find('.//{*}Counters').attrib
        counters.update(total=str(len(rows)),passed=str(sum(r.attrib['outcome']=='Passed' for r in rows)),failed=str(sum(r.attrib['outcome']=='Failed' for r in rows)))
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'result.trx';doc.write(path)
            evidence=Path(folder)/'evidence';evidence.mkdir()
            for key in ['backend','tensors']:
                (evidence/('selected-'+key+'.trx')).write_bytes((OLD/(key+'-tests')/(key+'.trx')).read_bytes())
            return (suite256 if disabled else suite)(path,name,evidence)

    def test_complete_declared_census(self):
        for name in ['backend','tensors']:
            for disabled in [False,True]:
                with self.subTest(name=name,disabled=disabled):
                    actual=self.check(self.fixture(name,disabled),name,disabled)
                    wanted=(3409,131) if name=='backend' and disabled else (3499,41) if name=='backend' else (368,0)
                    self.assertEqual((actual['passed'],actual['skipped']),wanted)

    def test_every_added_case_must_execute_once_under_its_exact_name(self):
        for name in ['backend']:
            for disabled in [False,True]:
                for mutation in ['remove','rename','skip','duplicate','fail']:
                    with self.subTest(name=name,disabled=disabled,mutation=mutation):
                        doc=self.fixture(name,disabled);results=doc.find('.//{*}Results')
                        row=next(r for r in results if r.attrib.get('testName')==NEW_CASES[name][0])
                        if mutation=='remove':results.remove(row)
                        elif mutation=='rename':row.attrib['testName']+=' unexpected'
                        elif mutation=='skip':row.attrib['outcome']='NotExecuted'
                        elif mutation=='duplicate':results.append(copy.deepcopy(row))
                        else:row.attrib['outcome']='Failed'
                        with self.assertRaises(AssertionError):self.check(doc,name,disabled)

    def test_existing_hardware_skip_must_remain(self):
        doc=self.fixture('backend',True)
        row=next(r for r in doc.findall('.//{*}UnitTestResult') if r.attrib['testName']=='Lokad.Onnx.Backend.Tests.Exp512Tests.ProbeHandlesExceptionals')
        self.assertEqual(row.attrib['outcome'],'NotExecuted');row.attrib['outcome']='Passed'
        with self.assertRaises(AssertionError):self.check(doc,'backend',True)

    def test_retained_scope(self):self.assertTrue(verify_scope())


if __name__=='__main__':unittest.main()
