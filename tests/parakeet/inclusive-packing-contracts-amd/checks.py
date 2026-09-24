"""Require the complete focused census and the intended selected-release failure."""
from collections import Counter
import xml.etree.ElementTree as ET
from protocol import pin, read

PREFIX = 'Lokad.Onnx.Backend.Tests.'
NEGATIVE = PREFIX + 'PackedBoundaryTests.Reduction4096_IsAdmitted'
IDENTITY = PREFIX + 'IdentityTests.ConsumedProductsAndInstructionModeMatch'


def contract(folder, spec, built, run):
    negative = folder.name == 'selected-negative'
    role = 'selected' if negative else 'candidate'
    xml = ET.parse(folder / 'contracts.trx')
    rows = xml.findall('.//{*}UnitTestResult')
    counts = xml.find('.//{*}Counters').attrib
    expected = {NEGATIVE: 1, IDENTITY: 1} if negative else spec['expected_cases']
    names = [r.attrib['testName'] for r in rows]
    assert len(set(names)) == len(rows) == int(counts['total']) == sum(expected.values())
    assert Counter(name.split('(')[0] for name in names) == expected
    assert int(counts['executed']) == len(rows)
    assert int(counts['passed']) == (1 if negative else 62)
    assert int(counts['failed']) == (1 if negative else 0)
    for row in rows:
        wanted = 'Failed' if negative and row.attrib['testName'] == NEGATIVE else 'Passed'
        assert row.attrib['outcome'] == wanted, row.attrib
        if wanted == 'Failed':
            message = row.find('.//{*}Message').text
            assert 'Assert.Equal() Failure' in message and 'Expected: 524288' in message and 'Actual:   0' in message
            assert 'Reduction4096_IsAdmitted' in row.find('.//{*}StackTrace').text
    value = read(folder / 'loaded.json')
    assert value['passed'] and str(value['pid']) in run['members']
    identities = spec['identities'][role]
    assert value['core_sha256'] == identities['Lokad.Onnx.dll']['sha256']
    assert value['data_sha256'] == identities['Lokad.Onnx.Data.dll']['sha256']
    assert value['consumer_sha256'] == built['consumer']['sha256']
    runtime = run['contract_environment']['PACKING_RUNTIME']
    assert runtime == '/dev/shm/lokad-parakeet-inclusive-packing-contracts-20260924/runtime/' + role
    assert value['core_path'] == runtime + '/Lokad.Onnx.dll'
    assert value['data_path'] == runtime + '/Lokad.Onnx.Data.dll'
    assert value['avx512'] == (not folder.name.endswith('-256')) and value['avx2'] and value['fma']
    assert value['runtime'] == '.NET 10.0.8' and value['affinity'] == 4 and value['processor_count'] == 1
    return dict(passed=True, tests=len(rows), passed_tests=int(counts['passed']),
                expected_failures=int(counts['failed']), skipped=0, role=role,
                avx512=value['avx512'], loaded=pin(folder / 'loaded.json'), trx=pin(folder / 'contracts.trx'))
