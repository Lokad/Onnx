"""Prove that native-close changes cannot bypass exact managed/public qualification."""
import copy
import hashlib
from pathlib import Path
import shutil
import tempfile
import unittest
import prepare  # Supplies the pinned local Python dependencies; starts no worker.
import numpy as np
from protocol import read,save
from checks import inventory,model
from qualify_outputs import pyannote

ROOT=Path(__file__).resolve().parents[3]
APP=ROOT/'artifacts/pyannote-blocked-spatial-app-amd-execution-20260922/collected/campaign'
PAYLOAD=ROOT/'artifacts/pyannote-blocked-spatial-app-amd-payload-20260922/payload'


class Checks(unittest.TestCase):
    def fixture(self,base):
        (base/'manifests').mkdir();shutil.copytree(PAYLOAD/'graph-reference',base/'graph-reference')
        shutil.copy2(PAYLOAD/'graph-reference.json',base/'graph-reference.json')
        for role in ['selected','candidate']:
            shutil.copy2(PAYLOAD/'manifests/portable-pyannote.json',base/'manifests'/(role+'-pyannote.json'))
            shutil.copytree(APP/'portable-pyannote-output',base/role/'output')
    def test_native_close_output_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);self.fixture(base)
            self.assertTrue(model(base,'candidate')['passed'])
            result=read(base/'candidate/output/result.json');name=result['rows'][0]['name']
            for row in result['rows']:
                if row['name']==name and row['model']=='segmentation':
                    path=base/'candidate/output'/row['output']['file'];values=np.fromfile(path,dtype='<f4')
                    values[0]=np.nextafter(values[0],np.float32(np.inf));values.tofile(path)
                    row['output']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            save(base/'candidate/output/result.json',result)
            self.assertTrue(pyannote(base,base/'candidate/output','candidate',base/'selected/output')['passed'])
            with self.assertRaises(AssertionError):model(base,'candidate')
    def test_public_and_missing_request_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);self.fixture(base);original=read(base/'candidate/output/result.json')
            bad=copy.deepcopy(original);bad['applications'].pop();save(base/'candidate/output/result.json',bad)
            with self.assertRaises(AssertionError):model(base,'candidate')
            bad=copy.deepcopy(original);bad['applications'][0]['result']['Intervals'][0]['Speaker']+=1
            save(base/'candidate/output/result.json',bad)
            with self.assertRaises(AssertionError):model(base,'candidate')
            bad=copy.deepcopy(original);bad['inputs_and_held_outputs_unchanged']=False;save(base/'candidate/output/result.json',bad)
            with self.assertRaises(AssertionError):model(base,'candidate')
    def test_only_identity_literal_changes(self):
        original=read(ROOT/'artifacts/pyannote-blocked-spatial-models-20260922/consumer-instructions.json');row=original['observations'][0]
        spec=dict(consumers=dict(selected=dict(sha256=row['before_sha256'])),old_data='85d166b59e2beef18ca7664f76faf445bf3cd81509f8f1d1c4b3c5354f53757a',new_data=prepare.OLD_DATA)
        built=dict(consumer=dict(sha256=row['after_sha256']))
        self.assertTrue(inventory(original,spec,built)['passed'])
        for change in ['extra-method','changed-body']:
            value=copy.deepcopy(original);r=value['observations'][0]
            if change=='extra-method':r['added'].append('Unexpected')
            else:r['candidate_methods'][r['differences'][0]]+='extra instruction'
            with self.assertRaises(AssertionError):inventory(value,spec,built)


if __name__=='__main__':unittest.main()
