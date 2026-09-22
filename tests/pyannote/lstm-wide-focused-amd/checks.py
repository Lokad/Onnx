"""Preserve every original test and require every new case to pass."""
import xml.etree.ElementTree as ET
from protocol import pin

PREFIX='Lokad.Onnx.Backend.Tests.LstmWideProjectionTests.'
NEW={PREFIX+'SingleAndStridedRowsKeepSelectedBitsAndSentinels(reduction: '+str(i)+')' for i in [0,1,13,60,128,256]}|{
    PREFIX+'NonfiniteAndSignedZeroMatchUnchangedPortableHelpers',PREFIX+'ExplicitExecutionModesKeepCompleteTrajectoriesAndOwnedOutputs'}
MESSAGE='System.InvalidOperationException : Tensor intrinsics were explicitly requested but x86 FMA is not supported on this machine.'


def parse(path):
    doc=ET.parse(path);items=doc.findall('.//{*}UnitTestResult');counts=doc.find('.//{*}Counters').attrib
    assert len(items)==int(counts['total'])==int(counts['executed'])
    rows={r.attrib['testName']:r for r in items};assert len(rows)==len(items)
    assert set(r.attrib['outcome'] for r in items)<= {'Passed','Failed'}
    assert sum(r.attrib['outcome']=='Passed' for r in items)==int(counts['passed'])
    assert sum(r.attrib['outcome']=='Failed' for r in items)==int(counts['failed'])
    return rows


def check_suite(path,ordinary,scalar,width):
    baseline=parse(ordinary);unsupported_reference=parse(scalar);rows=parse(path)
    assert len(baseline)==150 and set(baseline)==set(unsupported_reference)
    assert all(r.attrib['outcome']=='Passed' for r in baseline.values())
    unsupported={name for name,row in unsupported_reference.items() if row.attrib['outcome']=='Failed'}
    assert len(unsupported)==18 and all('LstmReferenceTests.MultipleBatchesDirectionsActivationsAndStorageMatchOrt(' in n and n.endswith('mode: 2)') for n in unsupported)
    for name in unsupported:
        row=unsupported_reference[name]
        assert row.find('.//{*}Message').text==MESSAGE and 'TensorExecutionOptions.Validate()' in row.find('.//{*}StackTrace').text
    assert len(rows)==158 and set(rows)==set(baseline)|NEW
    failed={name for name,row in rows.items() if row.attrib['outcome']=='Failed'}
    assert failed==(unsupported if width=='scalar' else set()),failed
    for name in failed:
        assert rows[name].find('.//{*}Message').text==MESSAGE and 'TensorExecutionOptions.Validate()' in rows[name].find('.//{*}StackTrace').text
    assert all(rows[name].attrib['outcome']=='Passed' for name in NEW)
    return dict(passed_supported=True,tests=158,passed=158-len(failed),unsupported_explicit_intrinsics=len(failed),
        new_passed=8,original_census_equal=True,trx=pin(path),no_performance_measurement=True)
