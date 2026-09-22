"""Fixed qualification/timing coverage, identities and resource bounds."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

GIB = 1024**3
ROLES = ('production', 'portable')
TIMING_ROLES = (*ROLES, 'ort', 'ort', *reversed(ROLES))
LIMITS = dict(worker_seconds=3600, campaign_seconds=4*3600, rss=12*GIB,
              available=GIB, preflight_available=12*GIB, public_preflight_available=14*GIB, tmpfs_free=GIB,
              preflight_tmpfs_free=3*GIB, artifact_bytes=2*GIB)
REQUIRED_TESTS = (
    'PackedAvx512RowTests.RowGroupsMatchIndependentFmaOracleAndPreserveGuards',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    with Path(path).open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def pin(path):
    path = Path(path)
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def below(root, name):
    path = (Path(root)/name).resolve()
    assert path != Path(root).resolve() and path.is_relative_to(Path(root).resolve()), name
    return path


def verified_files(root, files):
    for name, wanted in files.items():
        assert pin(below(root, name)) == wanted, name


def verify(base, external=True):
    spec = read(base/'payload.json')
    execution = read(base/'execution/execution.json')
    assert pin(base/'payload.json') == execution['payload']
    assert execution['limits'] == LIMITS
    verified_files(base, spec['files'])
    verified_files(base/'execution', execution['files'])
    if external:
        for name in spec['external'].keys() & execution['external'].keys():
            assert spec['external'][name] == execution['external'][name], name
        for name, wanted in dict(execution['external'], **spec['external']).items():
            assert pin(Path(name)) == wanted, name
    return spec, execution


def check_sample(sample):
    assert 0 <= sample['seconds'] < LIMITS['worker_seconds']
    assert sample['available'] >= LIMITS['available']
    assert sample['tmpfs_free'] >= LIMITS['tmpfs_free']
    assert sample['artifact_bytes'] <= LIMITS['artifact_bytes']
    assert sample['members'] and sum(m['rss'] for m in sample['members']) < LIMITS['rss']
    for member in sample['members']:
        assert member['rss'] >= 0 and member['affinity'] == [2]
        assert member['threads'] and all(t['affinity'] == [2] for t in member['threads'])


def test_results(path, minimum, required=()):
    root = ET.parse(path).getroot()
    ns = {'t': 'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
    counters = root.find('t:ResultSummary/t:Counters', ns)
    assert counters is not None
    rows = root.findall('t:Results/t:UnitTestResult', ns)
    assert len(rows) == int(counters.attrib['total'])
    assert all(r.attrib['outcome'] in ('Passed', 'NotExecuted') for r in rows)
    passed = sum(r.attrib['outcome'] == 'Passed' for r in rows)
    assert passed == int(counters.attrib['passed']) >= minimum
    for name in required:
        found = [r for r in rows if name in r.attrib['testName']]
        assert found and all(r.attrib['outcome'] == 'Passed' for r in found), ('Hardware test did not execute', name)
    return dict(passed=passed, skipped=[r.attrib['testName'] for r in rows if r.attrib['outcome'] == 'NotExecuted'],
                required_executed=list(required), trx=pin(path))


def gate(reports):
    """No failed or missing role is permitted to reach the timing stage."""
    assert set(reports) == set(ROLES)
    for role in ROLES:
        value = reports[role]
        assert value['pyannote']['passed'] is True
        assert value['pyannote']['arrays'] == 18 and value['pyannote']['public_calls'] == 16
        assert value['parakeet']['audit_consistent'] is True
        assert value['parakeet']['application_passed'] is True
        assert value['parakeet']['numeric_gate_passed'] is True
        assert value['parakeet']['arrays'] == 784 and value['parakeet']['values'] == 3090494
