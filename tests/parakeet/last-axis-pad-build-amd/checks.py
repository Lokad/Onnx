"""Reject collateral method changes and incomplete targeted-test results."""
from collections import Counter
import xml.etree.ElementTree as ET
from protocol import pin

PAD = 'Lokad.Onnx.CPUExecutionProvider::PadCore::Lokad.Onnx.DenseTensor`1[T] PadCore[T](Lokad.Onnx.Tensor`1[T], Int32[], Int32[], T, Boolean)'
HELPER = 'Lokad.Onnx.CPUExecutionProvider::TryPadLastAxis::Boolean TryPadLastAxis[T](Lokad.Onnx.DenseTensor`1[T], Lokad.Onnx.DenseTensor`1[T], Int32[])'
TESTS = [
    'FloatPaddingPreservesBitsAndOwnership',
    'DoublePaddingPreservesBitsAndOwnership',
    'Int32PaddingPreservesBitsAndOwnership',
    'Int64PaddingPreservesBitsAndOwnership',
    'SlicedAndBroadcastInputsRemainUnchanged',
    'ReflectionRetainsItsExistingPath',
]


def inventory(value, measured, built):
    assert value['inventory_complete'] and len(value['observations']) == 2
    for row, (name, count) in zip(value['observations'], [('Lokad.Onnx.dll', 3181), ('Lokad.Onnx.Data.dll', 697)], strict=True):
        assert row['assembly'] == name and row['methods'] == count
        assert row['before_sha256'] == measured[name]['sha256'] and row['after_sha256'] == built[name]['sha256']
        assert row['public_surface_equal'] and row['compiler_rename'] is None
        assert not row['removed'] and len(row['normalized_methods']) == count
        before, after = row['method_flags_before'], row['method_flags_after']
        assert set(before) == set(row['normalized_methods'])
        assert len(before) == count and all(after[k] == v for k, v in before.items())
        if name == 'Lokad.Onnx.dll':
            assert row['differences'] == [PAD] and row['added'] == [HELPER]
            assert row['unchanged_methods'] == 3180
            assert set(row['candidate_methods']) == {PAD, HELPER}
            assert row['candidate_methods'][PAD] != row['normalized_methods'][PAD]
            assert set(after) == set(before) | {HELPER} and after[HELPER] == 0
        else:
            assert row['unchanged_methods'] == 697
            assert not row['differences'] and not row['added'] and not row['candidate_methods']
            assert after == before
    return dict(passed=True, core_existing_methods=3181, core_unchanged_methods=3180,
                data_unchanged_methods=697, only_padcore_changed=True,
                added_private_helper=HELPER, public_surface_equal=True,
                existing_implementation_flags_equal=True, generated_names_exact=True,
                padcore_composition_review_pending=True)


def targeted_suite(path, avx512_disabled):
    root = ET.parse(path).getroot()
    rows = root.findall('.//{*}UnitTestResult')
    counts = root.find('.//{*}Counters').attrib
    assert (int(counts['total']), int(counts['passed']), int(counts['failed'])) == (6, 6, 0)
    expected = Counter(('Lokad.Onnx.Backend.Tests.LastAxisPadTests.' + name, 'Passed') for name in TESTS)
    actual = Counter((r.attrib['testName'], r.attrib['outcome']) for r in rows)
    assert actual == expected, dict(expected=list(expected), actual=list(actual))
    return dict(passed=6, skipped=0, census_exact=True, avx512_disabled=avx512_disabled, trx=pin(path))
