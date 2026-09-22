"""Shared evidence and resource checks for the normal-source test correction."""
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-blocked-spatial-composition-v2-20260922'
PRODUCT = ROOT/'artifacts/pyannote-blocked-spatial-composition-20260922'
FOCUSED = ROOT/'artifacts/pyannote-blocked-spatial-composition-review-v3-20260922'
FAILED = ROOT/'artifacts/pyannote-blocked-spatial-regressions-v2-20260922'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
FEED = ROOT/'artifacts/pyannote-amd-candidates-v3-20260921/payload/nuget-feed'
spec = importlib.util.spec_from_file_location('normal_source_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor); monitor.BASE = BASE
pin, read, save, verify, terminal = monitor.pin, monitor.read, monitor.save, monitor.verify, monitor.terminal


def resources(folder, state, numerical):
    assert state['complete']
    identities = [state['supervisor']]; outcomes = []
    for row in state['runs']:
        assert row['complete'] and row['seconds'] < 900
        assert row['preflight']['available'] >= (12 if row['name'] in numerical else 8)*1024**3
        samples = [json.loads(s) for s in (folder/'logs'/(row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0 and max(s['rss'] for s in samples) == row['peak_rss']
        for s in samples:
            assert s['seconds'] < 900 and s['rss'] < 8*1024**3 and s['available'] >= 1024**3
            assert s['disk'] >= 20*1024**3 and s['output_bytes'] <= 1024**3 and s['rss'] == sum(m['rss'] for m in s['members'])
            for m in s['members']: assert m['affinity'] == [2] and row['members'][str(m['pid'])] == m['birth']
        identities.extend(dict(pid=int(p), birth=b) for p, b in row['members'].items())
        outcomes.append(dict(name=row['name'], samples=len(samples), peak_rss=row['peak_rss']))
    for identity in identities: terminal(identity)
    return outcomes, identities


def suite(folder, name, passed, skipped, failed):
    tree = ET.parse(folder/'test-results'/(name+'.trx')); rows = tree.findall('.//{*}UnitTestResult')
    counts = tree.find('.//{*}Counters').attrib
    # xUnit records skipped rows as NotExecuted but leaves the summary counter 0.
    assert (int(counts['total']), int(counts['passed']), int(counts['executed']), int(counts['failed'])) == (passed+skipped+failed, passed, passed+failed, failed), counts
    assert len(rows) == passed+skipped+failed
    for outcome, count in [('Passed', passed), ('NotExecuted', skipped), ('Failed', failed)]:
        assert sum(r.attrib['outcome'] == outcome for r in rows) == count
    return dict(name=name, passed=passed, skipped=skipped, failed=failed, rows=[dict(r.attrib) for r in rows])


def close_failure():
    assert not (FAILED/'failure-closed.json').exists()
    value = read(FAILED/'inputs.json'); verify(value['files'])
    state = read(FAILED/'controller.json'); assert state['code'] == 1
    assert [(r['name'], r['code']) for r in state['runs']] == [('backend-existing', 0), ('tensors', 1)]
    observed, identities = resources(FAILED, state, {'backend-existing', 'tensors'})
    outcomes = [suite(FAILED, 'backend-existing', 3313, 93, 0), suite(FAILED, 'tensors', 342, 0, 1)]
    failed, = [r for r in outcomes[-1]['rows'] if r['outcome'] == 'Failed']
    assert failed['testName'] == 'Lokad.Onnx.Tensors.Tests.NoOptionalParametersTests.SourceTree_HasNoOptionalParameters'
    tree = ET.parse(FAILED/'test-results/tensors.trx')
    row, = [r for r in tree.findall('.//{*}UnitTestResult') if r.attrib['outcome'] == 'Failed']
    message = row.find('.//{*}Message').text
    assert message.startswith('Optional parameters found:')
    assert all(line.startswith('tests/Lokad.Onnx.Backend.Tests/ConvBlockedSpatialTests.cs: ') for line in message.splitlines()[1:])
    save(FAILED/'failure-analysis.json', dict(passed=False, retained_failure=True, suites=outcomes,
        resources=observed, failed_test=failed['testName'], message=message, product_arithmetic_failure=False))
    files = {p.relative_to(FAILED).as_posix(): pin(p) for p in FAILED.rglob('*') if p.is_file()}
    save(FAILED/'failure-closed.json', dict(passed=False, retained_failure=True, files=files, identities=identities, local_inputs=value['files']))


def priors():
    for folder, filename, passed in [(PRODUCT, 'failure-closed.json', False), (FOCUSED, 'closed.json', True), (FAILED, 'failure-closed.json', False)]:
        proof = read(folder/filename); assert proof['passed'] == passed
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
        for identity in proof.get('identities', []): terminal(identity)


def inventory():
    value = read(BASE/'instructions.json'); assert value['inventory_complete']
    assert [(r['assembly'], r['methods']) for r in value['observations']] == [('Lokad.Onnx.dll', 3161), ('Lokad.Onnx.Data.dll', 697)]
    for row in value['observations']:
        assert row['unchanged_methods'] == row['methods'] and row['public_surface_equal']
        assert not row['removed'] and not row['added'] and not row['differences']
        assert row['before_sha256'] == pin(PRODUCT/'runtime'/row['assembly'])['sha256']
        assert row['after_sha256'] == pin(BASE/'runtime'/row['assembly'])['sha256']
    return dict(passed=True, unchanged_core=3161, unchanged_data=697, public_surface_equal=True)
