"""Exercise scope guards with retained compiled IL, including deliberate damage."""
import copy
import json
from pathlib import Path
import unittest
from checks import consumer_scope, data_scope, layout

ROOT = Path(__file__).resolve().parents[3]
CORE_SHA = '37c243756bbe5e5d563e79627a0a4d20ff027598801849da9606549c7ac60286'


class ScopeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = ROOT/'artifacts/parakeet-managed-phase-amd-20260924/build-collected/inventory/instructions.json'
        if not path.exists(): raise unittest.SkipTest('Retained AMD IL inventory is required')
        inventory = json.loads(path.read_text())
        cls.methods = {}
        for row in inventory['observations']:
            key, = row['differences']
            # Only rename the two diagnostic types/variable and expected Core hash.
            # Instruction offsets, relative branches, exceptions and locals remain real.
            changed = row['candidate_methods'][key].replace('ParakeetPhaseProbe','ParakeetMaskingProbe') \
                .replace('PhaseConsumer','MaskingConsumer').replace('PARAKEET_PHASE_DATA_SHA','PARAKEET_MASKING_DATA_SHA') \
                .replace('672e5f303b011e27bb23097a49252c38ddee334938c75895e3c2341df0f3be35',CORE_SHA)
            cls.methods[row['assembly']] = (json.loads(row['normalized_methods'][key]),json.loads(changed))

    def test_preserves_actual_consumer_and_data_bodies(self):
        consumer_scope(*self.methods['SampledAudio.dll'],CORE_SHA)
        data_scope(*self.methods['Lokad.Onnx.Data.dll'])

    def test_rejects_removed_original_result_check(self):
        before, after = copy.deepcopy(self.methods['SampledAudio.dll'])
        check = next(r for r in after['instructions'] if 'g__Agree|' in r['operand'])
        check['operand'] = 'Bypass::Double Agree()'
        with self.assertRaises(AssertionError): consumer_scope(before,after,CORE_SHA)

    def test_rejects_branch_that_skips_an_original_check(self):
        before, after = copy.deepcopy(self.methods['SampledAudio.dll'])
        branch = next(r for r in after['instructions'] if r['opcode'] == 'beq.s')
        branch['operand'] = '00'  # Land on the following ldstr instead of the original success path.
        with self.assertRaises(AssertionError): consumer_scope(before,after,CORE_SHA)

    def test_rejects_changed_exception_region(self):
        before, after = copy.deepcopy(self.methods['SampledAudio.dll'])
        item = after['exceptions'][0]
        rows = after['instructions']; index = next(i for i,r in enumerate(rows) if r['offset'] == item['TryOffset'])
        delta = rows[index+1]['offset']-item['TryOffset']
        item['TryOffset'] += delta; item['TryLength'] -= delta
        with self.assertRaises(AssertionError): consumer_scope(before,after,CORE_SHA)

    def test_rejects_changed_original_graph_execution(self):
        before, after = copy.deepcopy(self.methods['Lokad.Onnx.Data.dll'])
        call = next(r for r in after['instructions'] if 'Boolean Execute(' in r['operand'])
        call['operand'] = 'Bypass::Boolean Execute()'
        with self.assertRaises(AssertionError): data_scope(before,after)

    def test_non_dense_layout_is_a_finding_but_invalid_storage_is_rejected(self):
        row = dict(Dimensions=[1,2,3],Strides=[6,1,2],Length=6,StorageLength=12,
            RuntimeType='Lokad.Onnx.TensorSlice`1[System.Single]',ExactDense=False,Reversed=True,
            ArrayBacked=True,ArrayOffset=4,ArrayCount=12,ArrayLength=16)
        found = layout(row)
        self.assertFalse(found['exact_dense']); self.assertFalse(found['row_major'])
        row['ArrayOffset'] = 5
        with self.assertRaises(AssertionError): layout(row)


if __name__ == '__main__': unittest.main()
