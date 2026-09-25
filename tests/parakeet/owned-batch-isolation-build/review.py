"""Reuse collection/resource checks; qualify only the declared dispatch relocation."""
import base64
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from run import BASE, BEFORE, REMOTE, TOOLS, PRELUDE, pin, read, write, ssh, prepared

loader = importlib.util.spec_from_file_location('parent_review', TOOLS.parent/'direct-depthwise-build-v2/review.py')
parent = importlib.util.module_from_spec(loader)
loader.loader.exec_module(parent)
collected = parent.collected


def build():
    assert not (BASE/'build-review.json').exists()
    folder, spec, receipt, state, built, resources = collected('build')
    assert (folder/'logs/sdk-version.stdout').read_text().strip() == '10.0.204'
    warnings = []
    for run in state['runs']:
        output = (folder/'logs'/(run['name']+'.stdout')).read_text()+(folder/'logs'/(run['name']+'.stderr')).read_text()
        assert ': error ' not in output
        warnings.extend(line for line in output.splitlines() if ': warning ' in line)
    baseline = read(BEFORE/'build-review.json')
    expected = [s.replace('/dev/shm/lokad-parakeet-direct-depthwise-build-v2-20260925', REMOTE) for s in baseline['warnings']]
    assert Counter(warnings) == Counter(expected), warnings
    inventory = read(folder/'logs/instructions.json')
    assert inventory['inventory_complete']
    assert [r['assembly'] for r in inventory['observations']] == ['Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll']
    methods = []
    for row in inventory['observations']:
        name = row['assembly']; core = name == 'Lokad.Onnx.dll'
        assert row['before_sha256'] == spec['before_product'][name]['sha256']
        assert row['after_sha256'] == built['product'][name]['sha256']
        assert row['methods'] == (3281 if core else 697)
        assert not row['added'] and not row['removed']
        assert len(row['differences']) == (3 if core else 0)
        assert {k.split('::')[1] for k in row['differences']} == ({'RunBatchedFloatMatMul', 'MatMulInto', 'MatMul'} if core else set())
        assert all(k.startswith('Lokad.Onnx.Tensor`1[T]::') for k in row['differences'])
        assert row['method_flags_before'] == row['method_flags_after']
        assert row['assembly_attributes_before'] == row['assembly_attributes_after']
        assert row['public_surface_equal'] and row['public_surface'] == row['public_surface_after']
        assert row['unchanged_methods'] == row['methods']-len(row['differences'])
        assert set(row['candidate_methods']) == set(row['differences'])
        if core:
            key, = [k for k in row['differences'] if '::RunBatchedFloatMatMul::' in k]
            assert inventory['release']['sha256'] == spec['release_product']['sha256']
            assert inventory['release']['methods'] == {key: row['candidate_methods'][key]}, 'Dispatcher must recover exact release instructions, locals and exceptions'
        else:
            assert built['product'][name] == spec['before_product'][name]
        methods.append(dict(assembly=name, original=row['methods'], unchanged=row['unchanged_methods'], changed=row['differences']))
    result = dict(passed=True, built=pin(folder/'built.json'), inventory=pin(folder/'logs/instructions.json'),
                  product=built['product'], consumer=built['consumer'], source=spec['source_prepared'],
                  methods=methods, warnings=warnings, zero_added_warnings=True, resources=resources,
                  release_dispatcher_restored=True, data_binary_unchanged=True, public_surface_unchanged=True,
                  release_admitted=False, failed_graph_cases=spec['failed_graph_cases'],
                  reviewer=pin(Path(__file__)), collection=pin(folder/'build-collection.json'))
    write(BASE/'build-review.json', result)
    encoded = base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    transferred = ssh(PRELUDE+f'''
from remote import verify,live,read,pin
import base64
verify();state=read(base/'build-state.json')
assert state['complete'] and state['code']==0 and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={result['built']!r}
with (base/'build-review.json').open('xb') as stream:stream.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert transferred['review'] == pin(BASE/'build-review.json')
    write(BASE/'build-review-transferred.json', transferred)
    print(json.dumps(dict(passed=True, methods=methods, product=built['product'], release_dispatcher_restored=True)))


def capture():
    assert not (BASE/'closed.json').exists()
    folder, spec, receipt, state, built, resources = collected('capture')
    assert pin(folder/'build-review.json') == pin(BASE/'build-review.json')
    assert read(BASE/'build-review.json')['built'] == pin(folder/'built.json')
    assert read(BASE/'build-collected/build-state.json')['ended'] < state['started']
    suites = []
    for mode, count in spec['expected_tests'].items():
        path = folder/'logs'/('contracts-'+mode+'.trx'); root = ET.parse(path).getroot()
        counters = root.find('.//{*}Counters'); assert counters is not None
        assert all(int(counters.get(k, '-1')) == count for k in ['total', 'executed', 'passed'])
        assert all(int(counters.get(k, '-1')) == 0 for k in ['failed', 'error', 'timeout', 'aborted', 'notExecuted', 'notRunnable'])
        results = root.findall('.//{*}UnitTestResult'); assert len(results) == count
        assert all(r.get('outcome') == 'Passed' for r in results)
        names = [r.get('testName') for r in results]; assert len(set(names)) == count
        by_class = Counter(n.split('(')[0].split('.')[-2] for n in names)
        expected = dict(MatMulDestinationTests=16, MatMulEmptyTests=4, MatMulVectorTests=3, OwnedPackedRuntimeIdentityTests=1)
        expected['OwnedPackedWeightTests' if mode == 'normal' else 'OwnedPackedUnavailableTests'] = 40 if mode == 'normal' else 1
        assert by_class == Counter(expected), by_class
        suites.append(dict(mode=mode, passed=count, skipped=0, classes=dict(by_class), names=names, trx=pin(path)))
    analysis = dict(passed=True, compiled_review=pin(BASE/'build-review.json'), product=built['product'],
                    consumer=built['consumer'], source=spec['source_prepared'], suites=suites, resources=resources,
                    release_admitted=False, no_application_score=True, graph_qualification_pending=True)
    write(BASE/'analysis.json', analysis)
    write(BASE/'closed.json', dict(passed=True, analysis=pin(BASE/'analysis.json'),
          compiled_review=pin(BASE/'build-review.json'), collection=pin(folder/'capture-collection.json'),
          transfer=pin(BASE/'capture-transfer.json'), terminal_owners=receipt['identities'], reviewer=pin(Path(__file__)),
          files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'), suites=[dict(mode=s['mode'],passed=s['passed']) for s in suites], product=built['product'])))


if __name__ == '__main__':
    {'build': build, 'capture': capture}[sys.argv[1]]()
