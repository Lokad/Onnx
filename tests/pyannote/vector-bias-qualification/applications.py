"""Apply the original suite/public-consumer qualification to the pooled Core."""
from common import *

BASE = ROOT / 'artifacts/pyannote-vector-bias-applications-20260921'
PRIOR = ROOT / 'artifacts/pyannote-convolution-portable-applications-20260921'
PUBLIC = ROOT / 'artifacts/pyannote-optimized-ort-20260921'
MEETINGS = ROOT / 'artifacts/pyannote-optimized-meetings-20260921'
INPUT = ROOT / 'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json'
ORIGINAL = ROOT / 'tests/pyannote/request-contexts/qualify.py'
PREREQUISITES = {
    ROOT / 'artifacts/pyannote-vector-bias-shared-20260921': '00f19df4287b5706a435f812b357c2e64efcd7cb57691970575e041f128f6cab',
    ROOT / 'artifacts/pyannote-vector-bias-parakeet-20260921': '51c3a0b861ab253054c42d7d3b4ad47200da882cf39d2672eed0140236d0ea1e',
    PRIOR: '20fd9f38b80f2f1e2b87a375b0b2c1c99e255d77f69b674271614361e7282002',
}


def prerequisites():
    proof = candidate()
    files = dict(proof['files'])
    files[rel(MODEL / 'closed.json')] = pin(MODEL / 'closed.json')
    for parent, sha in PREREQUISITES.items():
        path = parent / 'closed.json'
        assert pin(path)['sha256'] == sha
        item = read(path)
        assert item['passed']
        verify(item['files'])
        for identity in item.get('identities', item.get('terminal_identities', [])):
            terminal(identity)
        files.update(item['files'])
        files[rel(path)] = pin(path)
    for path in [ORIGINAL, Path(__file__), TOOLS / 'common.py', TOOLS / 'audit_applications.py']:
        files[rel(path)] = pin(path)
    BASE.mkdir()
    (BASE / 'logs').mkdir()
    monitor.BASE = BASE
    return dict(passed=True, files=files)


def load():
    source = ORIGINAL.read_text(encoding='utf8')
    def replace(old, new):
        nonlocal source
        assert source.count(old) == 1, old
        source = source.replace(old, new)
    replace("    assert pin(BASE / 'prepared.json')['sha256'] == '486c098dbc6c56b8e3f9dcfa7f5b478c98565dfe5d1d3713819b86b8a0c988e5'\n    prepared = read(BASE / 'prepared.json'); assert prepared['passed']; verify(prepared['files'])\n    builds = read(BASE / 'builds.json'); assert builds['complete'] and builds['code'] == 0; terminal(builds['supervisor'])",
            '    prepared = prerequisites()')
    replace("shutil.copytree(BASE / 'source', source,", "shutil.copytree(MODEL / 'source', source,")
    replace("runtime = BASE / 'runtime'", "runtime = MODEL / 'runtime'")
    replace("        assert count == (1 if name == 'tensors' else 2)",
            "        if name == 'backend':\n            assert count == 0\n            for n in dependencies:\n                assert '<HintPath>'+str(runtime / (n+'.dll'))+'</HintPath>' in before\n            changes.append(dict(path=path.relative_to(source).as_posix(), reason='Already references exact candidate runtime', diff=''))\n            continue\n        assert count == (1 if name == 'tensors' else 2)")
    replace("manifest.update(data_sha256=", "manifest.update(core_sha256=CORE, data_sha256=")
    replace('Request-scoped embedding contexts; unchanged qualified Core and original meeting consumer; no new native timing.',
            'Vectorized convolution bias with unchanged matrix kernels and Data; original meeting consumer; no new native timing.')
    namespace = dict(globals(), __name__='original_application_lane', __file__=str(Path(__file__)))
    exec(compile(source, str(ORIGINAL), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    load()['main']()
