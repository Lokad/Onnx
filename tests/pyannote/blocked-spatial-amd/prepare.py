"""Package the exact locally qualified consumer for both AMD instruction widths."""
import ast
import json
from pathlib import Path
import shutil
import sys
import tarfile
from protocol import LIMITS,pin,read,save

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-blocked-spatial-amd-20260922'
RAW=ROOT/'artifacts/pyannote-blocked-spatial-raw-v2-20260922'
PRIOR=ROOT/'artifacts/pyannote-selected-profile-amd-20260922'


def previous_closed():
    assert pin(PRIOR/'closed.json')['sha256']=='49dadc65ac0ec6ea63914377857549714f40fed87ca330be823ecc78fe3bd056'
    closed=read(PRIOR/'closed.json');assert closed['passed']
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    sys.path.insert(0,str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
    import psutil
    for identity in closed['local_identities']:
        try:assert psutil.Process(identity['pid']).create_time()!=identity['birth']
        except psutil.NoSuchProcess:pass
    collection=read(PRIOR/'collected/collection.json');assert collection['terminal'] and collection['input_error'] is None
    state=read(PRIOR/'collected/identity.json');assert state['complete'] and state['code']==0
    return state['supervisor']


def prepare():
    assert not BASE.exists();owner=previous_closed()
    assert pin(RAW/'closed.json')['sha256']=='943d8dc7d25fa95c9778e5ce4d62efbf693b9c63c4a2eebf04bb9370593d25b9'
    proof=read(RAW/'closed.json');assert proof['passed'] and proof['qualification_complete']
    for name,wanted in proof['files'].items():assert pin(ROOT/name)==wanted,name
    import psutil
    for identity in proof['identities']:
        try:assert psutil.Process(identity['pid']).create_time()!=identity['birth']
        except psutil.NoSuchProcess:pass
    payload=BASE/'payload';payload.mkdir(parents=True);(payload/'tools').mkdir()
    for name in ['remote.py','protocol.py']:shutil.copy2(TOOLS/name,payload/'tools'/name)
    shutil.copytree(RAW/'source/bin/Release/net10.0',payload/'runtime')
    (payload/'source').mkdir()
    for p in (RAW/'source').iterdir():
        if p.is_file():shutil.copy2(p,payload/'source'/p.name)
    shutil.copy2(RAW/'output/256.json',payload/'windows-256.json')
    shutil.copy2(ROOT/'.agent/m13-pyannote-blocked-spatial-20260922.md',payload/'prospective-plan.md')
    original=read(PRIOR/'payload/payload.json')
    spec=dict(passed=True,limits=LIMITS,previous_owner=owner,boot_time=1789634288.0,
        interpreter=original['external']['/usr/bin/python3'],external=original['external'],
        core=pin(payload/'runtime/Lokad.Onnx.dll'),consumer=pin(payload/'runtime/BlockedSpatialProbe.dll'),
        files={p.relative_to(payload).as_posix():pin(p) for p in payload.rglob('*') if p.is_file()},
        scope='Raw/layout/public-caller AMD numerical qualification in both AVX2 and AVX-512; no component timing')
    save(payload/'payload.json',spec)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in [RAW/'closed.json',PRIOR/'closed.json',*TOOLS.glob('*')] if p.is_file()}
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(payload.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(payload).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=files,payload=pin(payload/'payload.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(payload=pin(payload/'payload.json'),archive=pin(BASE/'payload.tar.gz'),files=len(spec['files']))))


if __name__=='__main__':prepare()
