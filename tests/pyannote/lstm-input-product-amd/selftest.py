"""A full test census and exact package identity must survive adversarial checks."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET
from checks import suite,package,consumer,HARDWARE
from protocol import pin,read

ROOT=Path(__file__).resolve().parents[3]
APP=ROOT/'artifacts/pyannote-blocked-spatial-app-amd-execution-20260922/collected/campaign/test-results'
PACKAGE=ROOT/'artifacts/pyannote-blocked-spatial-package-20260922'


class Checks(unittest.TestCase):
    def test_census_and_missing_hardware(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)
            for name in ['backend','tensors']:shutil.copy2(APP/(name+'.trx'),p/('selected-'+name+'.trx'))
            focused=ROOT/'artifacts/pyannote-lstm-input-blocks-v2-20260922/test-results/lstm-ordinary.trx';shutil.copy2(focused,p/'candidate-lstm.trx')
            doc=ET.parse(APP/'backend.trx');results=doc.find('.//{*}Results');counts=doc.find('.//{*}Counters')
            for row in ET.parse(focused).findall('.//{*}UnitTestResult'):
                if 'LstmInputBlockTests.' in row.attrib['testName']:results.append(copy.deepcopy(row))
            for key in ['total','executed','passed']:counts.attrib[key]=str(int(counts.attrib[key])+36)
            path=p/'complete.trx';doc.write(path)
            self.assertEqual(suite(path,'backend',p)['passed'],3432)
            hardware=next(r for r in results if HARDWARE in r.attrib['testName'])
            hardware.attrib['testName']='UnrelatedPassingTest';doc.write(path)
            with self.assertRaises(AssertionError):suite(path,'backend',p)
            self.assertEqual(suite(APP/'tensors.trx','tensors',p)['passed'],343)
    def test_package_and_consumer_identity(self):
        core=pin(PACKAGE/'consumer/bin/Release/net10.0/Lokad.Onnx.dll')
        self.assertTrue(package(PACKAGE/'nuget/Lokad.Onnx.0.2.0.nupkg',core)['passed'])
        with self.assertRaises(AssertionError):package(PACKAGE/'nuget/Lokad.Onnx.0.2.0.nupkg',dict(core,sha256='0'*64))
        result=read(PACKAGE/'consumer.json');result['runtime']='.NET 10.0.8';result['processor_count']=1
        exe=dict(sha256=result['executable']);model=dict(sha256=result['model'])
        self.assertTrue(consumer(result,core,exe,model)['passed'])
        for key,value in [('core','0'*64),('prepared_graph_calls',1),('graph_scratch',0),('input_and_held_outputs_unchanged',False)]:
            bad=dict(result);bad[key]=value
            with self.assertRaises(AssertionError):consumer(bad,core,exe,model)


if __name__=='__main__':unittest.main()
