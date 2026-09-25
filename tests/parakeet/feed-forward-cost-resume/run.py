"""Finish two diagnostic roles after retaining the original trace-cap failure."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
OLD_TOOLS = TOOLS.parent / 'feed-forward-cost-diagnostic'
sys.path.insert(0, str(OLD_TOOLS))
loader = importlib.util.spec_from_file_location('original_cost_run', OLD_TOOLS / 'run.py')
original = importlib.util.module_from_spec(loader)
loader.loader.exec_module(original)
pin, read, write, ssh = original.pin, original.read, original.write, original.ssh
OLD, OLD_REMOTE = original.BASE, original.REMOTE
OUTPUT_LIMIT = 2 * 1024**3


def paths(mode):
    assert mode in ['stages', 'markers']
    return (ROOT / f'artifacts/parakeet-feed-forward-cost-{mode}-resume-20260925',
            f'/dev/shm/lokad-parakeet-feed-forward-cost-{mode}-resume-20260925')


def initial():
    original.prepared()
    folder = OLD / 'capture-collected'
    receipt = read(folder / 'capture-collection.json')
    transfer = read(OLD / 'capture-transfer.json')
    state = read(folder / 'capture-state.json')
    assert pin(folder / 'capture-collection.json')['sha256'] == '12023f7ce722110aa0c47817d12b75bf63aa6e70019e133af37985a0618782da'
    assert transfer['passed'] and transfer['archive'] == pin(OLD / 'capture-results.tar.gz')
    assert transfer['collection'] == pin(folder / 'capture-collection.json')
    assert receipt['terminal'] and receipt['code'] == 1 and receipt['state'] == pin(folder / 'capture-state.json')
    assert state['complete'] and state['code'] == 1 and state['supervisor'] == read(OLD / 'capture-deployment.json')
    for name, wanted in receipt['files'].items():
        assert pin(folder / name) == wanted, name
    assert [r['name'] for r in state['runs']] == ['clock', 'stages']
    clock, stopped = state['runs']
    assert clock['complete'] and clock['code'] == 0 and len(read(folder / 'clock/result.json')['records']) == 80
    assert stopped['complete'] and stopped['code'] == -9 and stopped['samples'] == 210
    assert len(list((folder / 'stages').glob('cost-*.json'))) == 33
    assert not (folder / 'stages/result.json').exists() and not (folder / 'markers').exists()
    spec = read(OLD / 'bundle/spec.json')
    samples = [json.loads(s) for s in (folder / 'logs/stages.resources.jsonl').read_text().splitlines()]
    assert all(s['output'] < spec['output_limit'] for s in samples[:-1])
    assert samples[-1]['output'] == 542543106 >= spec['output_limit']
    for s in samples:
        assert s['rss'] < spec['capture_limits']['rss'] and s['seconds'] < spec['capture_limits']['seconds']
        assert min(s['available'], s['tmpfs']) >= spec['minimum_free']
    failure = ROOT / 'artifacts/parakeet-feed-forward-cost-audit-failure-20260925.json'
    assert read(failure)['code'] == 1
    review = OLD / 'build-review.json'
    assert pin(review)['sha256'] == '951250373bd21764a19a4944b5869c6539a33f4df2990c643960dab883b4d21c'
    traces = sorted((folder / 'stages').glob('cost-*.json'))
    return dict(collection=pin(folder / 'capture-collection.json'), transfer=pin(OLD / 'capture-transfer.json'),
        state=pin(folder / 'capture-state.json'), failed_audit=pin(failure), build_review=pin(review),
        first_corpus_bytes=sum(p.stat().st_size for p in traces[:20]),
        maximum_observed_request_bytes=max(p.stat().st_size for p in traces),
        original_code=1, original_clock_repeated=False, partial_stage_requests_retained=33)


def worker():
    source = (OLD_TOOLS / 'vm.py').read_text(encoding='utf8')
    edits = [
        ("approval = common.read(BASE / 'build-review.json')", "approval = common.read(Path(spec['original']) / 'build-review.json')"),
        ("approval['built'] == common.pin(BASE / 'built.json')", "approval['built'] == common.pin(Path(spec['original']) / 'built.json')"),
        ("built = common.read(BASE / 'built.json')", "built = common.read(Path(spec['original']) / 'built.json')"),
        ("common.pin(BASE / name) == wanted, name", "common.pin(Path(spec['original']) / name) == wanted, name"),
        ("for mode in MODES:", "for mode in [spec['mode']]:"),
        ("runtime = BASE / ('runtime-observed' if mode == 'markers' else 'runtime-control')", "runtime = Path(spec['original']) / ('runtime-observed' if mode == 'markers' else 'runtime-control')"),
        ("assert [r['name'] for r in state['runs']] == MODES", "assert [r['name'] for r in state['runs']] == [spec['mode']]"),
    ]
    changed = source
    for before, after in edits:
        assert changed.count(before) == 1, before
        changed = changed.replace(before, after)
    restored = changed
    for before, after in reversed(edits):
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored == source
    ast.parse(changed)
    return changed


def prepare(mode):
    base, remote = paths(mode)
    assert not base.exists()
    first = initial()
    base.mkdir(); bundle = base / 'bundle'; bundle.mkdir()
    (bundle / 'remote.py').write_text(worker(), encoding='utf8')
    (bundle / 'common.py').write_bytes((OLD / 'bundle/common.py').read_bytes())
    (bundle / 'recovery.md').write_bytes((TOOLS / 'README.md').read_bytes())
    write(bundle / 'initial.json', first)
    spec = read(OLD / 'bundle/spec.json')
    # No workload, runtime, acceptance, memory or CPU setting changes.
    spec.update(mode=mode, original=OLD_REMOTE, initial=first, output_limit=OUTPUT_LIMIT)
    external = spec['external']
    external.update({OLD_REMOTE + '/' + n: v for n, v in read(OLD / 'bundle/spec.json')['files'].items()})
    external.update({OLD_REMOTE + '/' + n: v for n, v in read(OLD / 'capture-collected/built.json')['runtime_files'].items()})
    for name in ['spec.json', 'built.json', 'build-review.json', 'capture-state.json', 'capture-collection.json']:
        external[OLD_REMOTE + '/' + name] = pin(OLD / 'capture-collected' / name)
    spec['files'] = {p.relative_to(bundle).as_posix(): pin(p) for p in bundle.iterdir()}
    write(bundle / 'spec.json', spec)
    with tarfile.open(base / 'payload.tar.gz', 'w:gz') as archive:
        for p in bundle.iterdir():
            archive.add(p, arcname=p.name, recursive=False)
    for p in TOOLS.glob('*.py'):
        ast.parse(p.read_text(encoding='utf8'), str(p))
    write(base / 'prepared.json', dict(archive=pin(base / 'payload.tar.gz'), spec=pin(bundle / 'spec.json'),
        initial=first, original_preparation=pin(OLD / 'prepared.json'),
        tools={p.name: pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(mode=mode, prepared=True, archive=pin(base / 'payload.tar.gz'), output_limit=OUTPUT_LIMIT)))


def prepared(mode):
    base, remote = paths(mode)
    value = read(base / 'prepared.json')
    assert value['initial'] == initial() and value['original_preparation'] == pin(OLD / 'prepared.json')
    assert value['archive'] == pin(base / 'payload.tar.gz') and value['spec'] == pin(base / 'bundle/spec.json')
    for name, wanted in value['tools'].items():
        assert pin(TOOLS / name) == wanted, name
    spec = read(base / 'bundle/spec.json'); prior = read(OLD / 'bundle/spec.json')
    excluded = {'files', 'external', 'mode', 'original', 'initial', 'output_limit'}
    assert {k:v for k,v in spec.items() if k not in excluded} == {k:v for k,v in prior.items() if k not in excluded}
    assert spec['output_limit'] == OUTPUT_LIMIT and spec['mode'] == mode and spec['original'] == OLD_REMOTE
    assert all(spec['external'][n] == v for n,v in prior['external'].items())
    for name, wanted in spec['files'].items():
        assert pin(base / 'bundle' / name) == wanted
    assert (base / 'bundle/remote.py').read_text(encoding='utf8') == worker()
    return spec


def action(mode, name):
    prepared(mode)
    base, remote = paths(mode)
    prelude = original.PRELUDE.replace(OLD_REMOTE, remote)
    if name == 'launch':
        assert read(base / 'staged.json')['passed']
        # Keep each complete role only once; release VM trace copies in between.
        prior = 'original' if mode == 'stages' else 'stages'
        assert read(ROOT / f'artifacts/parakeet-feed-forward-cost-{prior}-trace-retention-20260925/closed.json')['passed']
    if name in ['stage', 'launch']:
        transport = original.transport
        transport.BASE, transport.REMOTE, transport.PRELUDE = base, remote, prelude
        return transport.stage() if name == 'stage' else transport.launch('capture')
    # Reuse the complete original collector and compact live-handle observer.
    original.BASE, original.REMOTE, original.PRELUDE = base, remote, prelude
    assert name in ['observe', 'collect']
    return getattr(original, name)('capture')


if __name__ == '__main__':
    command, mode = sys.argv[1:]
    if command == 'prepare':
        prepare(mode)
    else:
        action(mode, command)
