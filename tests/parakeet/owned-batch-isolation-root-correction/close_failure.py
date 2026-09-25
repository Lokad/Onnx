"""Preserve the exact root source-policy failure without admitting the release."""
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT/'tests/parakeet/owned-batch-isolation-root-amd'
sys.path.insert(0,str(ORIGINAL))
from run import BASE, prepared
from protocol import JOBS, pin, read, save, check_sample
from checks import inventory, suite


def main():
    spec = prepared(); assert not (BASE/'closed.json').exists()
    folder = BASE/'collected'
    receipt = read(folder/'collection.json'); transfer = read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['receipt'] == pin(folder/'collection.json')
    assert transfer['archive'] == pin(BASE/'results.tar.gz')
    assert receipt['terminal'] and receipt['code'] == 1 and receipt['input_error'] is None
    assert receipt['payload'] == pin(BASE/'payload.json')
    for name,wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
    state = read(folder/'identity.json')
    assert state['complete'] and state['code'] == 1
    assert [r['name'] for r in state['runs']] == JOBS[:10]
    assert all(r['complete'] and r['code'] == (1 if r['name'] == 'tensors-tests' else 0) for r in state['runs'])
    resources = []
    for row in state['runs']:
        samples = [json.loads(s) for s in (folder/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples) == row['samples'] > 0
        for sample in samples:
            check_sample(sample)
            assert all(row['members'][str(m['pid'])] == m['birth'] for m in sample['members'])
        resources.append(dict(name=row['name'],samples=len(samples),peak_rss=max(s['rss'] for s in samples)))
    payload = read(BASE/'payload.json'); built = read(folder/'built.json')
    product = inventory(read(folder/'inventory/instructions.json'),payload['measured'],built['product'])
    backend = suite(folder/'backend-tests/backend.trx','backend',folder/'evidence')
    trx = ET.parse(folder/'tensors-tests/tensors.trx')
    counts = trx.find('.//{*}Counters').attrib
    assert (int(counts['total']),int(counts['passed']),int(counts['failed'])) == (394,392,2)
    failures = {r.attrib['testName']:r.find('.//{*}Message').text for r in trx.findall('.//{*}UnitTestResult') if r.attrib['outcome'] == 'Failed'}
    expected = {
        'Lokad.Onnx.Tensors.Tests.ImplementorDocsTests.SourceTree_DeclaresNoFriendAssemblies',
        'Lokad.Onnx.Tensors.Tests.NoOptionalParametersTests.SourceTree_HasNoOptionalParameters'}
    assert set(failures) == expected
    assert failures[next(n for n in expected if 'ImplementorDocsTests' in n)] == 'Unapproved InternalsVisibleTo found in:\nsrc/Lokad.Onnx/Global.cs'
    optional = failures[next(n for n in expected if 'NoOptionalParametersTests' in n)].splitlines()
    assert optional[0] == 'Optional parameters found:' and len(optional) == 10
    assert all(line.startswith(('tests/Lokad.Onnx.Backend.Tests/OwnedPackedWeightTests.cs:',
        'tests/Lokad.Onnx.Backend.Tests/DirectDepthwiseTests.cs:')) for line in optional[1:])
    analysis = dict(passed=False,admitted=False,execution_complete=True,known_source_policy_failures=failures,
        inventory=product,backend=backend,tensors=dict(total=394,passed=392,failed=2),
        remaining_jobs_not_run=JOBS[10:],resources=resources,measured=payload['measured'],built=built['product'],
        original_guards_unchanged=True,correction_not_applied=True,no_performance_measurement=True)
    save(BASE/'analysis.json',analysis)
    save(BASE/'closed.json',dict(passed=False,admitted=False,preserved_failure=True,
        analysis=pin(BASE/'analysis.json'),remote_terminal=receipt['identities'],
        local_inputs=spec['files'],files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        auditor=pin(Path(__file__))))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**analysis)))


if __name__ == '__main__': main()
