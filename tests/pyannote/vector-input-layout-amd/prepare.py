"""Freeze actual-shape operands and the exact Windows-qualified consumer."""
import ast
import io
import json
from pathlib import Path
import shutil
import sys
import tarfile
import unittest
from protocol import LIMITS, pin, read, save
from score import iteration_manifest

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-vector-input-layout-amd-20260922'
PRIOR = ROOT/'artifacts/pyannote-vector-output-epilogue-amd-20260922'
MODEL = ROOT/'artifacts/pyannote-vector-input-layout-20260922'
FIXTURES = ROOT/'artifacts/pyannote-blocked-spatial-fixtures-20260922'


def closed(folder, expected):
    assert pin(folder/'closed.json')['sha256'] == expected
    proof = read(folder/'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
    return proof


def previous_closed():
    closed(PRIOR, 'd5dedc5f486c218804acd1d8b560dd80c7c2a390dcb256907b60ec5d64099049')
    receipt = read(PRIOR/'collected/collection.json'); assert receipt['terminal'] and receipt['input_error'] is None
    return read(PRIOR/'deployment.json')


def prepare():
    assert not BASE.exists(); owner = previous_closed()
    model = closed(MODEL, '8dae29462e8d0694031b41e0db31387da550b0796c2fc386ebecf2f00e21a978')
    fixtures = closed(FIXTURES, '27ac5ab76c21828d18e59478f6010a4139df0530b349ba5863e5f771dc97422e')
    sys.path.insert(0, str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
    import psutil
    for identity in model['identities'] + fixtures['identities']:
        try: assert psutil.Process(identity['pid']).create_time() != identity['birth']
        except psutil.NoSuchProcess: pass
    assert psutil.virtual_memory().available >= 12*1024**3 and shutil.disk_usage(ROOT).free >= 20*1024**3
    payload = BASE/'payload'; payload.mkdir(parents=True); (payload/'tools').mkdir(); (payload/'source').mkdir()
    suite = unittest.defaultTestLoader.discover(str(TOOLS), pattern='test_score.py')
    output = io.StringIO(); result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    save(BASE/'selftest.json', dict(passed=result.wasSuccessful(), tests=result.testsRun, output=output.getvalue()))
    assert result.wasSuccessful() and result.testsRun == 12
    for name in ['remote.py', 'protocol.py']: shutil.copy2(TOOLS/name, payload/'tools'/name)
    shutil.copytree(MODEL/'source/bin/Release/net10.0', payload/'runtime')
    for p in (MODEL/'source').iterdir():
        if p.is_file(): shutil.copy2(p, payload/'source'/p.name)
    (payload/'fixtures').mkdir()
    result = read(FIXTURES/'output/result.json')
    for name in ['result.json', *[r['path'] for r in result['tensors'].values()]]:
        shutil.copy2(FIXTURES/'output'/name, payload/'fixtures'/name)
    save(payload/'fixtures/iterations.json', iteration_manifest(result['calls']))
    shutil.copy2(MODEL/'output/model-256.json', payload/'windows-256.json')
    shutil.copy2(MODEL/'output/raw-256.json', payload/'windows-raw.json')
    shutil.copy2(ROOT/'.agent/m16-pyannote-vector-input-20260922.md', payload/'prospective-plan.md')
    previous = read(PRIOR/'payload/payload.json')
    spec = dict(passed=True, limits=LIMITS, previous_owner=owner, boot_time=1789634288.0,
        interpreter=previous['interpreter'], external=previous['external'],
        core=pin(payload/'runtime/Lokad.Onnx.dll'), consumer=pin(payload/'runtime/VectorInputProbe.dll'),
        files={p.relative_to(payload).as_posix(): pin(p) for p in payload.rglob('*') if p.is_file()},
        scope='Raw and actual-model qualification in both widths, then one fixed complete-component screen of the vector input successor.')
    save(payload/'payload.json', spec)
    files = {p.relative_to(ROOT).as_posix(): pin(p) for p in [MODEL/'closed.json', FIXTURES/'closed.json', PRIOR/'closed.json', *TOOLS.iterdir()] if p.is_file()}
    for p in TOOLS.glob('*.py'): ast.parse(p.read_text(), str(p))
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for p in sorted(payload.rglob('*')):
            if p.is_file(): archive.add(p, arcname=p.relative_to(payload).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=files, payload=pin(payload/'payload.json'), archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(payload=pin(payload/'payload.json'), archive=pin(BASE/'payload.tar.gz'), files=len(spec['files']))))


if __name__ == '__main__': prepare()
