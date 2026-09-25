"""Exercise the scope checker on retained real IL; reject extra observer changes."""
import copy
import json
import unittest
from prepare_source import PHASE,pin,read
from compiled_scope import verify_data


class CompiledScope(unittest.TestCase):
    def setUp(self):
        path=PHASE/'build-collected/inventory/instructions.json'
        assert read(PHASE/'build-review.json')['inventory']==pin(path)
        self.reference,=[r for r in read(path)['observations'] if r['assembly']=='Lokad.Onnx.Data.dll']
        self.row=copy.deepcopy(self.reference)

    def verify(self):
        return verify_data(self.row,self.reference['before_sha256'],self.reference['after_sha256'],self.reference)

    def test_existing_certified_scope_passes(self):
        self.assertTrue(self.verify()['observer_helpers_exact'])

    def test_constructor_change_is_rejected(self):
        self.row['differences'].append('Lokad.Onnx.ParakeetTranscriber::.ctor::changed')
        self.row['unchanged_methods']-=1
        with self.assertRaises(AssertionError):self.verify()

    def test_changed_execute_instruction_is_rejected(self):
        key,=self.row['differences'];body=json.loads(self.row['candidate_methods'][key])
        body['instructions'][3]['opcode']='nop'
        self.row['candidate_methods'][key]=json.dumps(body)
        with self.assertRaises(AssertionError):self.verify()

    def test_changed_dispose_operand_is_rejected(self):
        key,=self.row['differences'];body=json.loads(self.row['candidate_methods'][key])
        row=next(i for i in body['instructions'] if i['offset']>=120 and i['opcode']=='callvirt')
        row['operand']='Wrong::Dispose()';self.row['candidate_methods'][key]=json.dumps(body)
        with self.assertRaises(AssertionError):self.verify()

    def test_changed_helper_is_rejected(self):
        self.row['candidate_methods'][self.row['added'][0]]+=' '
        with self.assertRaises(AssertionError):self.verify()

    def test_changed_flags_are_rejected(self):
        key=next(iter(self.row['method_flags_before']))
        self.row['method_flags_after'][key]='changed'
        with self.assertRaises(AssertionError):self.verify()

    def test_wrong_original_product_is_rejected(self):
        self.row['before_sha256']='0'*64
        with self.assertRaises(AssertionError):self.verify()


if __name__=='__main__':unittest.main()
