"""Collect two omitted input directories without changing frozen graph workers."""
import ast
import difflib
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT/'tests/parakeet/observed-dense-where-graphs-amd'
BASE = ROOT/'artifacts/parakeet-observed-dense-where-graphs-amd-20260924'
REVIEW = ROOT/'artifacts/parakeet-observed-dense-where-graph-collection-correction-20260924'
BEFORE = "[*JOBS,'logs','runtimes','previous','bridge','reference','evidence','tools']"
AFTER = "[*JOBS,'logs','runtimes','runtimes-e5','source','previous','bridge','reference','evidence','tools']"


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def text_pin(value):
    raw = value.encode('utf8')
    return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def scope():
    prepared = read(BASE/'prepared.json')
    original = TOOLS/'run.py'
    assert pin(original) == prepared['files'][original.relative_to(ROOT).as_posix()]
    assert pin(original)['sha256'] == '9154f93996d742e07ee9331e5034250791faa52babbe1daf9ff4e386428fd593'
    assert pin(BASE/'payload.json') == read(BASE/'staged.json')['payload']
    payload = read(BASE/'payload.json')
    sys.path.insert(0,str(TOOLS))
    spec = importlib.util.spec_from_file_location('frozen_graph_transport',original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    before = inspect.getsource(module.collect)
    assert before.count(BEFORE) == 1 and AFTER not in before
    after = before.replace(BEFORE,AFTER)
    assert after.replace(AFTER,BEFORE) == before
    tree = ast.parse(after)
    assert len(tree.body) == 1 and isinstance(tree.body[0],ast.FunctionDef) and tree.body[0].name == 'collect'
    compile(after,str(__file__)+':frozen-collect-with-input-directories','exec')
    prior_dirs = {'logs','runtimes','previous','bridge','reference','evidence','tools'}
    payload_dirs = {n.split('/')[0] for n in payload['files'] if '/' in n}
    assert payload_dirs-prior_dirs == {'runtimes-e5','source'}
    extras = {n:v for n,v in payload['files'].items() if n.split('/')[0] in {'runtimes-e5','source'}}
    assert len(extras) == 11 and sum(n.startswith('runtimes-e5/') for n in extras) == 10
    assert [n for n in extras if n.startswith('source/')] == ['source/global.json']
    for role in ['current','candidate']:
        assert extras[f'runtimes-e5/{role}/ReleaseBenchmark.dll'] == payload['e5_consumer']
        assert extras[f'runtimes-e5/{role}/Lokad.Onnx.dll'] == payload['products'][role]['Lokad.Onnx.dll']
    review = dict(passed=True,collector=pin(Path(__file__)),frozen_transport=pin(original),
        payload=pin(BASE/'payload.json'),original_function=text_pin(before),corrected_function=text_pin(after),
        added_directories=['runtimes-e5','source'],added_inputs=extras,
        original_inputs_removed=False,worker_or_scoring_changed=False,
        reason='Frozen collector omits the separate qualified e5 runtime required by the audit and source/global.json. Add only these input directories; retain all original collection and terminal/hash checks.')
    patch = ''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
        fromfile='frozen run.py::collect',tofile='in-memory collection correction'))
    return module,after,review,patch


def review():
    assert not REVIEW.exists()
    _,_,value,patch = scope()
    REVIEW.mkdir()
    (REVIEW/'collection.patch').write_text(patch,encoding='utf8')
    value['patch'] = pin(REVIEW/'collection.patch')
    (REVIEW/'review.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(passed=True,review=pin(REVIEW/'review.json'),
        added_files=len(value['added_inputs']),added_bytes=sum(v['bytes'] for v in value['added_inputs'].values()),
        worker_or_scoring_changed=False)))


def collect():
    module,after,value,patch = scope()
    expected = read(REVIEW/'review.json')
    assert expected == dict(**value,patch=pin(REVIEW/'collection.patch'))
    assert (REVIEW/'collection.patch').read_text(encoding='utf8') == patch
    assert not (BASE/'collected').exists() and not (BASE/'results.tar.gz').exists()
    receipt_path = BASE/'collection-adapter.json'
    assert not receipt_path.exists()
    # Existing prepare, terminal-owner, payload and transfer checks remain exact.
    module.prepared()
    receipt_path.write_text(json.dumps(dict(review=pin(REVIEW/'review.json'),**expected),indent=2)+'\n',encoding='utf8')
    exec(compile(after,str(__file__)+':frozen-collect-with-input-directories','exec'),module.__dict__)
    module.collect()
    receipt = read(BASE/'collected/collection.json')
    for name,wanted in expected['added_inputs'].items():
        assert receipt['files'][name] == wanted == pin(BASE/'collected'/name),name
    print(json.dumps(dict(collection_correction_verified=True,added_files=len(expected['added_inputs']),
        receipt=pin(receipt_path))))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['review','collect']
    globals()[sys.argv[1]]()
