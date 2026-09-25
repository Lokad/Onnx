"""Exercise exact retained lineage and reject hidden baseline/public changes."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from protocol import pin,read,save
from lineage import qualify_lineage

ROOT=Path(__file__).resolve().parents[3]
BEFORE=ROOT/'artifacts/parakeet-slice-dense-conversion-models-amd-20260925'
AFTER=ROOT/'artifacts/parakeet-packed-final-row-models-amd-20260925'


class LineageTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.base=Path(self.temporary.name)
        old=read(BEFORE/'analysis.json');new=read(AFTER/'analysis.json')
        self.reports={'parakeet-release':old,'parakeet':new,'product':copy.deepcopy(new)}
        self.spec=dict(identities=dict(selected=copy.deepcopy(old['identities']['selected']),
                                       candidate=copy.deepcopy(new['identities']['candidate'])))
        for generation,folder in [('release',BEFORE),('m78',AFTER)]:
            for role in ['selected','candidate']:
                for isa in ['512','256']:
                    path=self.base/'evidence/public-lineage'/generation/(role+'-'+isa+'.json')
                    path.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(folder/'collected'/(role+'-public-'+isa)/'output/result.json',path)

    def verify(self):return qualify_lineage(self.base,self.reports,self.spec)

    def change_public(self,edit):
        path=self.base/'evidence/public-lineage/m78/selected-256.json'
        value=read(path);edit(value);save(path,value)
        self.reports['parakeet']['results']['selected-public-256']['result']=pin(path)

    def test_exact_retained_outputs_and_products_pass(self):
        result=self.verify();self.assertTrue(result['passed']);self.assertEqual(len(result['results']),8)
        self.assertEqual(result['identities'],self.spec['identities'])

    def test_relabeling_intermediate_as_release_fails(self):
        self.spec['identities']['selected']=copy.deepcopy(self.reports['parakeet']['identities']['selected'])
        with self.assertRaises(AssertionError):self.verify()

    def test_different_intermediate_data_fails(self):
        self.reports['parakeet']['identities']['selected']['Lokad.Onnx.Data.dll']['sha256']='0'*64
        with self.assertRaises(AssertionError):self.verify()

    def test_missing_last_tensor_fails(self):
        self.reports['parakeet']['results']['candidate-native-256']['native']['exact_selected_comparisons'].pop()
        with self.assertRaises(AssertionError):self.verify()

    def test_changed_tensor_hash_fails_even_with_success_flags(self):
        self.reports['parakeet']['results']['candidate-native-512']['native']['exact_selected_comparisons'][-1]['sha256']='0'*64
        with self.assertRaises(AssertionError):self.verify()

    def test_complete_public_result_drift_fails(self):
        self.change_public(lambda v:v['records'][-1]['result'].update(unexpected=True))
        with self.assertRaises(AssertionError):self.verify()

    def test_input_drift_fails(self):
        self.change_public(lambda v:v['records'][-1].update(input_sha256='0'*64))
        with self.assertRaises(AssertionError):self.verify()

    def test_duplicate_clip_fails(self):
        self.change_public(lambda v:v['records'][-1].update(name=v['records'][0]['name']))
        with self.assertRaises(AssertionError):self.verify()


if __name__=='__main__':unittest.main()
