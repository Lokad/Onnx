"""Exercise rejection of collateral changes and missing/skipped Pad coverage."""
import copy
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from checks import HELPER, PAD, TESTS, inventory, targeted_suite


class GateTests(unittest.TestCase):
    def fixture(self):
        products = {name: dict(bytes=1, sha256=name) for name in ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']}
        rows = []
        for name, count in [('Lokad.Onnx.dll', 3181), ('Lokad.Onnx.Data.dll', 697)]:
            core = name == 'Lokad.Onnx.dll'
            methods = {str(i): 'old-body' for i in range(count - int(core))}
            if core:
                methods[PAD] = 'old-pad'
            flags = {key: 0 for key in methods}
            rows.append(dict(assembly=name, methods=count, before_sha256=name, after_sha256=name,
                public_surface_equal=True, compiler_rename=None, removed=[], normalized_methods=methods,
                method_flags_before=flags, method_flags_after=dict(flags, **{HELPER: 0}) if core else dict(flags),
                differences=[PAD] if core else [], added=[HELPER] if core else [],
                unchanged_methods=count-int(core), candidate_methods={PAD: 'new-pad', HELPER: 'new-helper'} if core else {}))
        return dict(inventory_complete=True, observations=rows), products

    def test_only_declared_scope_passes(self):
        value, products = self.fixture()
        self.assertTrue(inventory(value, products, products)['only_padcore_changed'])

    def test_rejects_each_collateral_change(self):
        value, products = self.fixture()
        mutations = [
            lambda r: r['method_flags_after'].__setitem__('0', 512),
            lambda r: r.__setitem__('public_surface_equal', False),
            lambda r: r.__setitem__('compiler_rename', {'oldKey': 'a', 'newKey': 'b'}),
            lambda r: r['differences'].append('0'),
            lambda r: r['removed'].append('0'),
            lambda r: r['added'].append('extra-helper'),
            lambda r: r['candidate_methods'].__setitem__(PAD, r['normalized_methods'][PAD]),
            lambda r: r['method_flags_after'].__setitem__(HELPER, 512),
            lambda r: r['method_flags_after'].__setitem__('extra-flag', 0),
            lambda r: r.__setitem__('before_sha256', 'wrong-binary'),
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutations.index(mutation)):
                changed = copy.deepcopy(value)
                mutation(changed['observations'][0])
                with self.assertRaises(AssertionError):
                    inventory(changed, products, products)

    def test_rejects_data_change(self):
        value, products = self.fixture()
        value['observations'][1]['differences'].append('0')
        with self.assertRaises(AssertionError):
            inventory(value, products, products)

    def test_targeted_suite_requires_all_six_names_and_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'pad.trx'
            for mode in ['valid', 'missing', 'skipped', 'duplicate', 'wrong-name']:
                root = ET.Element('TestRun')
                results = ET.SubElement(root, 'Results')
                for i, name in enumerate(TESTS):
                    if mode == 'missing' and i == 0:
                        continue
                    if mode == 'duplicate' and i == 0:
                        name = TESTS[1]
                    if mode == 'wrong-name' and i == 0:
                        name = 'UnrelatedTest'
                    ET.SubElement(results, 'UnitTestResult', testName='Lokad.Onnx.Backend.Tests.LastAxisPadTests.' + name,
                        outcome='NotExecuted' if mode == 'skipped' and i == 0 else 'Passed')
                ET.SubElement(root, 'Counters', total='6', passed='6', failed='0')
                ET.ElementTree(root).write(path)
                if mode == 'valid':
                    self.assertEqual(targeted_suite(path, False)['passed'], 6)
                else:
                    with self.subTest(mode=mode), self.assertRaises(AssertionError):
                        targeted_suite(path, False)


if __name__ == '__main__':
    unittest.main()
