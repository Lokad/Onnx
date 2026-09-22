"""Freeze actual-shape operands and the exact Windows-qualified consumer."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tarfile
from protocol import LIMITS, pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-blocked-spatial-model-amd-20260922'
PRIOR = ROOT/'artifacts/pyannote-blocked-spatial-amd-20260922'
MODEL = ROOT/'artifacts/pyannote-blocked-spatial-model-20260922'
FIXTURES = ROOT/'artifacts/pyannote-blocked-spatial-fixtures-20260922'


def closed(folder, expected):
    assert pin(folder/'closed.json')['sha256'] == expected
    proof = read(folder/'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
    return proof


def previous_closed():
    closed(PRIOR, 'f2ac8145a31e9d4801bca1ac0e13d18825ea87c409eb8bae4ca15dad19445c92')
    receipt = read(PRIOR/'collected/collection.json'); assert receipt['terminal'] and receipt['input_error'] is None
    return read(PRIOR/'deployment.json')


def prepare():
    assert not BASE.exists(); owner = previous_closed()
    model = closed(MODEL, 'bb8f483c6dc6edad3ed6f89f739ccc1fadf7c8cd67ec507e24b617d73c53f54d')
    fixtures = closed(FIXTURES, '27ac5ab76c21828d18e59478f6010a4139df0530b349ba5863e5f771dc97422e')
    sys.path.insert(0, str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
    import psutil
    for identity in model['identities'] + fixtures['identities']:
        try: assert psutil.Process(identity['pid']).create_time() != identity['birth']
        except psutil.NoSuchProcess: pass
    assert psutil.virtual_memory().available >= 12*1024**3 and shutil.disk_usage(ROOT).free >= 20*1024**3
    payload = BASE/'payload'; payload.mkdir(parents=True); (payload/'tools').mkdir(); (payload/'source').mkdir()
    for name in ['remote.py', 'protocol.py']: shutil.copy2(TOOLS/name, payload/'tools'/name)
    shutil.copytree(MODEL/'source/bin/Release/net10.0', payload/'runtime')
    for p in (MODEL/'source').iterdir():
        if p.is_file(): shutil.copy2(p, payload/'source'/p.name)
    (payload/'fixtures').mkdir()
    result = read(FIXTURES/'output/result.json')
    for name in ['result.json', *[r['path'] for r in result['tensors'].values()]]:
        shutil.copy2(FIXTURES/'output'/name, payload/'fixtures'/name)
    shutil.copy2(MODEL/'output/256.json', payload/'windows-256.json')
    shutil.copy2(ROOT/'.agent/m13-pyannote-blocked-spatial-20260922.md', payload/'prospective-plan.md')
    previous = read(PRIOR/'payload/payload.json')
    spec = dict(passed=True, limits=LIMITS, previous_owner=owner, boot_time=1789634288.0,
        interpreter=previous['interpreter'], external=previous['external'],
        core=pin(payload/'runtime/Lokad.Onnx.dll'), consumer=pin(payload/'runtime/BlockedSpatialModel.dll'),
        files={p.relative_to(payload).as_posix(): pin(p) for p in payload.rglob('*') if p.is_file()},
        scope='Both AMD widths, all 108 actual-shape calls. Numerical qualification only.')
    save(payload/'payload.json', spec)
    files = {p.relative_to(ROOT).as_posix(): pin(p) for p in [MODEL/'closed.json', FIXTURES/'closed.json', PRIOR/'closed.json', *TOOLS.iterdir()] if p.is_file()}
    for p in TOOLS.glob('*.py'): ast.parse(p.read_text(), str(p))
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for p in sorted(payload.rglob('*')):
            if p.is_file(): archive.add(p, arcname=p.relative_to(payload).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=files, payload=pin(payload/'payload.json'), archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(payload=pin(payload/'payload.json'), archive=pin(BASE/'payload.tar.gz'), files=len(spec['files']))))


if __name__ == '__main__': prepare()
