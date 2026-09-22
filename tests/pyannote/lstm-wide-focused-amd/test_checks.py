"""Reject missing, skipped, failing, renamed and falsely counted test results."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from checks import NEW,check_suite

ROOT=Path(__file__).resolve().parents[3]
ORDINARY=ROOT/'artifacts/pyannote-lstm-input-blocks-amd-20260922/collected/suite-512/suite.trx'
SCALAR=ROOT/'artifacts/pyannote-lstm-input-blocks-v3-20260922/test-results/lstm-scalar.trx'


class Results(unittest.TestCase):
    def test_rejects_census_and_outcome_mutations(self):
        def document(original):
            tree=ET.parse(original);container=tree.find('.//{*}Results');example=container.find('{*}UnitTestResult')
            for name in sorted(NEW):
                row=ET.SubElement(container,example.tag,dict(example.attrib,testName=name,outcome='Passed'))
            counts=tree.find('.//{*}Counters');counts.set('total','158');counts.set('executed','158');counts.set('passed',str(int(counts.get('passed'))+8))
            return tree
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'result.trx';base=document(ORDINARY);base.write(path)
            self.assertEqual(check_suite(path,ORDINARY,SCALAR,'512')['passed'],158)
            scalar=document(SCALAR);scalar.write(path)
            self.assertEqual(check_suite(path,ORDINARY,SCALAR,'scalar')['passed'],140)
            mutations=[lambda tree:tree.find('.//{*}Counters').set('passed','157'),
                lambda tree:tree.find('.//{*}UnitTestResult').set('outcome','NotExecuted'),
                lambda tree:tree.find('.//{*}UnitTestResult').set('testName','renamed'),
                lambda tree:tree.find('.//{*}Results').remove(tree.find('.//{*}UnitTestResult'))]
            for mutate in mutations:
                damaged=deepcopy(base);mutate(damaged);damaged.write(path)
                with self.assertRaises(AssertionError):check_suite(path,ORDINARY,SCALAR,'512')
            scalar.write(path)
            with self.assertRaises(AssertionError):check_suite(path,ORDINARY,SCALAR,'512')
            base.write(path)
            with self.assertRaises(AssertionError):check_suite(path,ORDINARY,SCALAR,'scalar')


if __name__=='__main__':unittest.main()
