"""Apply the original suite/public-consumer qualification to the pooled Core."""
from common import *

BASE = ROOT / 'artifacts/pyannote-convolution-portable-applications-20260921'
PRIOR = ROOT / 'artifacts/pyannote-convolution-pool-applications-v2-20260921'
PUBLIC = ROOT / 'artifacts/pyannote-optimized-ort-20260921'
MEETINGS = ROOT / 'artifacts/pyannote-optimized-meetings-20260921'
INPUT = ROOT / 'artifacts/audio-ort-baseline-v2-20260919/inputs/pyannote.json'
ORIGINAL = ROOT / 'tests/pyannote/request-contexts/qualify.py'
PREREQUISITES = {
    ROOT / 'artifacts/pyannote-convolution-portable-shared-20260921': '32f84f780f9fbb74080ee7414a1e477f92bb9cd1fe287620ed50b7e833bb4282',
    ROOT / 'artifacts/pyannote-convolution-portable-parakeet-20260921': '56ceec578ca673b36ab685c456c248d8fd09810fc561c69f7ccc24dc72fa2751',
    PRIOR: 'e3fa3f17df5ec4731bc75ebf519389c571cc4f8d9537d6098f9e9f4ce0a28a7a',
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
            'Portable convolution row grouping with unchanged raw kernels and Data; original meeting consumer; no new native timing.')
    namespace = dict(globals(), __name__='original_application_lane', __file__=str(Path(__file__)))
    exec(compile(source, str(ORIGINAL), 'exec'), namespace)
    return namespace


if __name__ == '__main__':
    load()['main']()
