"""Retain the failed root run as a failure; never invoke the success auditor."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parents[2]
sys.path.insert(1, str(TOOLS.parent/'pad-current-root-amd'))
from protocol import JOBS, pin, read, save

BASE = ROOT/'artifacts/parakeet-attention-owned-root-amd-20260928'
FAILURE = 'Lokad.Onnx.Tensors.Tests.NoOptionalParametersTests.SourceTree_HasNoOptionalParameters'


def verify():
    collected = BASE/'collected'
    receipt = read(collected/'collection.json')
    transfer = read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz')
    assert transfer['receipt'] == pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code'] == 1 and receipt['input_error'] is None
    assert receipt['payload'] == pin(BASE/'payload.json')
    for name, wanted in receipt['files'].items(): assert pin(collected/name) == wanted, name
    state = read(collected/'identity.json')
    assert state['complete'] and state['code'] == 1
    assert state['supervisor'] == read(BASE/'deployment.json')
    assert [r['name'] for r in state['runs']] == JOBS[:12]
    assert all(r['complete'] for r in state['runs'])
    assert [r['code'] for r in state['runs']] == [0]*11 + [1]
    assert receipt['identities'] == [state['supervisor']] + [
        dict(pid=int(p), birth=b) for r in state['runs'] for p, b in r['members'].items()]
    counts = {}
    for name, expected in [('backend', (3597, 43, 0)), ('tensors', (393, 0, 1))]:
        rows = ET.parse(collected/(name+'-tests')/(name+'.trx')).findall('.//{*}UnitTestResult')
        observed = tuple(sum(r.attrib['outcome'] == status for r in rows)
                         for status in ['Passed', 'NotExecuted', 'Failed'])
        assert observed == expected and len(rows) == sum(expected)
        counts[name] = dict(passed=observed[0], skipped=observed[1], failed=observed[2])
        if name == 'tensors':
            failed, = [r for r in rows if r.attrib['outcome'] == 'Failed']
            assert failed.attrib['testName'] == FAILURE
            message = failed.find('.//{*}Message').text
            assert 'Optional parameters found:' in message
            assert message.count('OwnedAttentionPreparationTests.cs: Graph(') == 5
    inventory = read(collected/'inventory/review.json')
    assert inventory['passed'] and inventory['method_bodies_equal']
    assert (inventory['core_methods'], inventory['data_methods']) == (3288, 697)
    stage = read(BASE/'bundle/stage.json')
    for name, wanted in stage['files'].items(): assert pin(BASE/'bundle'/name) == wanted, name
    return dict(passed=False, release_admitted=False, terminal=True, code=1,
        reason='New portable test helper violates unchanged no-optional-parameters source policy',
        failed_test=FAILURE, suites=counts, unexecuted_jobs=JOBS[12:],
        compiled_scope=inventory, collection=pin(collected/'collection.json'),
        source_stage=pin(BASE/'bundle/stage.json'), terminal_owners=receipt['identities'])


if __name__ == '__main__':
    assert not (BASE/'failed.json').exists() and not (BASE/'closed.json').exists()
    result = verify()
    result['files'] = {p.relative_to(BASE).as_posix(): pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'failed.json', result)
    print(dict(failure=pin(BASE/'failed.json'), failed_test=result['failed_test'], suites=result['suites']))
