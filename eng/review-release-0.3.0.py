"""Audit the collected release checks; this does not build, publish or query CI."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/release-readiness-20260928'
FINAL = BASE/'v030'
COLLECTED = FINAL/'collected'
NS = {'t':'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def census(folder):
    result = {}
    for mode in ['normal','scalar']:
        files = sorted((folder/('results-'+mode)).glob('*.trx')); assert len(files) == 2
        tests = {}
        for path in files:
            tree = ET.parse(path).getroot()
            counts = tree.find('t:ResultSummary/t:Counters', NS).attrib
            assert all(int(counts[k]) == 0 for k in ['failed','error','timeout','aborted'])
            rows = tree.findall('t:Results/t:UnitTestResult', NS)
            assert len(rows) == int(counts['total'])
            for row in rows:
                name, outcome, identity = row.attrib['testName'], row.attrib['outcome'], row.attrib['testId']
                # TRX truncates long theory argument displays. Test IDs keep
                # those distinct cases instead of merging their names.
                assert identity not in tests and outcome in ['Passed','NotExecuted']
                tests[identity] = dict(name=name, outcome=outcome)
        assert len(tests) == 4040
        result[mode] = tests
    return result


def main():
    assert not (FINAL/'analysis.json').exists(), 'Preserve the completed audit'
    inputs = read(FINAL/'release-inputs.json'); assert inputs['version'] == '0.3.0'
    for name, expected in inputs['overlay'].items(): assert pin(ROOT/name) == expected, name
    transfer = read(FINAL/'transfer.json'); receipt = read(COLLECTED/'collection.json')
    assert transfer['passed'] and transfer['archive'] == pin(FINAL/'collected.tar.gz')
    assert transfer['collection'] == pin(COLLECTED/'collection.json') and receipt['terminal'] and receipt['code'] == 0
    for name, expected in receipt['files'].items(): assert pin(COLLECTED/name) == expected, name
    state = read(COLLECTED/'state.json'); deployment = read(FINAL/'deployment.json')
    assert all(state[k] == deployment[k] for k in ['pid','birth'])
    assert state['complete'] and state['code'] == 0
    assert [r['name'] for r in state['runs']] == ['restore','build','tests-normal','tests-scalar','pack-smoke','package-version','compiled-scope']
    resources = []
    for run in state['runs']:
        assert run['complete'] and run['code'] == 0 and run['seconds'] < 900
        samples = [json.loads(line) for line in (COLLECTED/(run['name']+'.resources.jsonl')).read_text().splitlines()]
        assert samples
        for sample in samples:
            assert sample['seconds'] < 900 and sample['rss'] < 8*1024**3
            assert min(sample['available'], sample['tmpfs']) > 1024**3
        resources.append(dict(name=run['name'],seconds=run['seconds'],peak_rss=max(r['rss'] for r in samples)))
    tests = census(COLLECTED)
    assert tests == census(BASE/'collected'), 'Final-version outcomes must preserve every corrected test case'
    counts = {mode:dict(Counter(row['outcome'] for row in rows.values())) for mode, rows in tests.items()}
    assert counts == dict(normal=dict(Passed=3997,NotExecuted=43), scalar=dict(Passed=3709,NotExecuted=331))
    for rows in tests.values():
        named = {row['name']:row['outcome'] for row in rows.values()}
        for name in ['SourceTree_HasNoOptionalParameters','ArchiveRecognitionRejectsChangedContentAndOtherPaths']:
            assert named['Lokad.Onnx.Tensors.Tests.NoOptionalParametersTests.'+name] == 'Passed'
        assert named['Lokad.Onnx.Backend.Tests.EncoderFoundationEdgeTests.PadCropsAndFillsUsingCoordinatesWithoutChangingInt64Bits(layout: 2)'] == 'Passed'
        reversed_lstm = [row['outcome'] for row in rows.values() if 'LstmReferenceTests.MultipleBatchesDirectionsActivationsAndStorageMatchOrt' in row['name'] and 'mode: 2)' in row['name']]
        assert len(reversed_lstm) == 18 and set(reversed_lstm) == {'Passed'}
    package = COLLECTED/'source/artifacts/nuget/Lokad.Onnx.0.3.0.nupkg'
    with zipfile.ZipFile(package) as archive:
        metadata = ET.fromstring(archive.read('Lokad.Onnx.nuspec'))
        ns = {'n':metadata.tag.split('}')[0].strip('{')}
        assert metadata.find('n:metadata/n:version', ns).text == '0.3.0'
        dependencies = metadata.findall('.//n:dependency', ns)
        assert [(d.attrib['id'],d.attrib['version']) for d in dependencies] == [('Google.Protobuf','3.33.5')]
        assert [n for n in archive.namelist() if n.startswith('lib/') and n.endswith('.dll')] == ['lib/net10.0/Lokad.Onnx.dll']
        for name in ['README.md','CHANGELOG.md','LICENSE.txt']:
            assert archive.read(name).replace(b'\r\n', b'\n') == (ROOT/name).read_bytes().replace(b'\r\n', b'\n'), name
        assert archive.read('icon.png') == (ROOT/'icon.png').read_bytes()
        assert archive.read('lib/net10.0/Lokad.Onnx.dll') == (COLLECTED/'package-runtime/Lokad.Onnx.dll').read_bytes()
    assert (COLLECTED/'package-version.stdout').read_text().strip() == '0.3.0.0'
    smoke = (COLLECTED/'pack-smoke.stdout').read_text()
    assert all(marker in smoke for marker in ['PASS pack','PASS contents','PASS consume','PASS smoke-pack'])
    identity = read(COLLECTED/'package-identity.json')
    assert identity['package_sha256'] == pin(package)['sha256']
    assert identity['core_sha256'] == pin(COLLECTED/'package-runtime/Lokad.Onnx.dll')['sha256']
    assert identity['core_sha256'] == identity['test_core_sha256'], 'Tests must use the packaged Core bytes'
    inventory = read(COLLECTED/'instructions.json'); assert inventory['inventory_complete']
    scope = []
    for row in inventory['observations']:
        expected = 3288 if row['assembly'] == 'Lokad.Onnx.dll' else 697
        assert row['methods'] == row['unchanged_methods'] == len(row['normalized_methods']) == expected
        assert not row['removed'] and not row['added'] and not row['differences'] and not row['candidate_methods']
        assert row['public_surface_equal'] and row['public_surface'] == row['public_surface_after']
        assert row['method_flags_before'] == row['method_flags_after']
        before = row['assembly_attributes_before']
        if row['assembly'] == 'Lokad.Onnx.dll':
            before = [a.replace('VersionAttribute("0.2.0")','VersionAttribute("0.3.0")') for a in before]
        assert sorted(before) == sorted(row['assembly_attributes_after'])
        assert row['after_sha256'] == pin(COLLECTED/'package-runtime'/row['assembly'])['sha256']
        scope.append(dict(assembly=row['assembly'],methods_unchanged=expected,public_surface_equal=True,flags_equal=True,
            before_sha256=row['before_sha256'],after_sha256=row['after_sha256']))
    assert {r['assembly'] for r in scope} == {'Lokad.Onnx.dll','Lokad.Onnx.Data.dll'}
    value = dict(passed=True,version='0.3.0',base_commit=inputs['base_commit'],package=pin(package),scope=scope,
        test_counts=counts,full_test_outcomes=tests,resources=resources,package_core_equals_tested_core=True,
        hosted_ci_green=False,hosted_ci_commit='5bc38b6cccdfe39873bed56be4daef2db355c42f',
        hosted_ci_url='https://github.com/Lokad/Onnx/actions/runs/36394756408',
        hosted_ci_failures=read(BASE/'current-failures.json'),inputs=inputs['overlay'],
        collection=pin(COLLECTED/'collection.json'),auditor=pin(Path(__file__)))
    (FINAL/'analysis.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    (FINAL/'closed.json').write_text(json.dumps(dict(passed=True,analysis=pin(FINAL/'analysis.json'),
        package=value['package'],transfer=pin(FINAL/'transfer.json'),auditor=value['auditor'],
        terminal_supervisor=deployment),indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:value[k] for k in ['passed','version','package','scope','test_counts','hosted_ci_green']},indent=2))


if __name__ == '__main__': main()
