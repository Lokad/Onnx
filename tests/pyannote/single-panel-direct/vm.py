"""Deploy the new, locally qualified component with the unchanged AMD protocol."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
LOCAL = ROOT / 'artifacts/pyannote-single-panel-direct-20260922'
BASE = ROOT / 'artifacts/pyannote-single-panel-amd-20260922'
PRIOR = ROOT / 'artifacts/pyannote-two-column-20260922'
REMOTE = '/dev/shm/lokad-pyannote-single-panel-20260922'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


common = module('single_panel_common', ROOT / 'tests/pyannote/direct-output/common.py')
common.BASE = BASE; common.REMOTE = REMOTE
pin, read, save, verify, terminal, rel = common.pin, common.read, common.save, common.verify, common.terminal, common.rel


def prepared():
    proof = read(LOCAL / 'closed.json')
    assert pin(LOCAL / 'closed.json')['sha256'] == '77450563eba3ddab07a1c81be4a1921f9c514a700490e1d2cfb90885866afd4f'
    assert proof['passed']; verify(proof['files'])
    for identity in proof['identities']: terminal(identity)
    value = read(BASE / 'prepared.json'); assert value['passed']; verify(value['files'])
    assert pin(BASE / 'payload.tar.gz') == value['archive']
    assert pin(BASE / 'payload/payload.json') == value['payload']
    return value


common.prepared = prepared
sys.modules['common'] = common
transport = module('single_panel_transport', ROOT / 'tests/pyannote/direct-output/transport.py')


def prepare():
    assert not BASE.exists()
    proof = read(LOCAL / 'closed.json'); verify(proof['files'])
    assert pin(LOCAL / 'closed.json')['sha256'] == '77450563eba3ddab07a1c81be4a1921f9c514a700490e1d2cfb90885866afd4f'
    for identity in proof['identities']: terminal(identity)
    previous = ROOT / 'artifacts/pyannote-direct-amd-execution-v2-20260922'
    controller = read(previous / 'controller/state.json'); assert controller['complete'] and controller['code'] == 0
    for identity in [controller['supervisor']] + [r['child'] for r in controller['stages']]: terminal(identity)
    assert pin(previous / 'closed.json')['sha256'] == 'ee5f73d8c286728316fe14ef7032ef42fa8fcd89a51be467f954303cfb595537'
    assert not read(previous / 'analysis.json')['performance']['admitted']
    BASE.mkdir(); payload = BASE / 'payload'; payload.mkdir()
    shutil.copytree(LOCAL / 'runtime', payload / 'runtime')
    shutil.copy2(LOCAL / 'shapes.json', payload / 'shapes.json')
    (payload / 'tools').mkdir()
    shutil.copy2(PRIOR / 'payload/tools/remote.py', payload / 'tools/remote.py')
    spec = read(PRIOR / 'payload/payload.json')
    assert spec['jobs'] == ['validate', 'validate-scalar-tail', 'baseline-a', 'candidate-a', 'candidate-b', 'baseline-b']
    spec.update(consumer=pin(payload / 'runtime/DirectOutputProbe.dll'), scalar_consumer=pin(payload / 'runtime/ScalarTailProbe.dll'),
                files={p.relative_to(payload).as_posix(): pin(p) for p in payload.rglob('*') if p.is_file()})
    save(payload / 'payload.json', spec)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as archive:
        for p in sorted(payload.rglob('*')):
            if p.is_file(): archive.add(p, arcname=p.relative_to(payload).as_posix(), recursive=False)
    files = {rel(p): pin(p) for p in [*TOOLS.glob('*'), LOCAL / 'closed.json', previous / 'closed.json',
             ROOT / 'tests/pyannote/direct-output/common.py', ROOT / 'tests/pyannote/direct-output/transport.py'] if p.is_file()}
    files.update({rel(p): pin(p) for p in payload.rglob('*') if p.is_file()})
    save(BASE / 'prepared.json', dict(passed=True, files=files, archive=pin(BASE / 'payload.tar.gz'), payload=pin(payload / 'payload.json')))
    prepared()
    print(json.dumps(dict(payload=pin(payload / 'payload.json'), archive=pin(BASE / 'payload.tar.gz'))))


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'stage', 'launch', 'observe', 'collect']
    if sys.argv[1] == 'prepare': prepare()
    else: getattr(transport, sys.argv[1])()
