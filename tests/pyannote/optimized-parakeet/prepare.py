"""Freeze complete Parakeet trajectories for current and optimized pyannote cores."""
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-optimized-parakeet-20260921'
OLD=ROOT/'artifacts/parakeet-transcription-20260919/frozen'
REFERENCE=OLD/'reference/manifest.json'
MODELS=ROOT/'models/parakeet-tdt-0.6b-v3'
PRODUCT=ROOT/'artifacts/whisper-memory-product-v2-20260921/source/src/Lokad.Onnx.CLI/bin/Release/net10.0'
CANDIDATE=ROOT/'artifacts/pyannote-lstm-output-lanes-20260921'


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def clean_env():return {k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}


def main():
    receipt=CANDIDATE/'qualification-closed.json'
    assert pin(receipt)['sha256']=='cf950ec5cedf702c1af38decc377cd516a5d8b652f77853d03d2c81db0b5bf53'
    closure=json.loads(receipt.read_text());assert closure['passed']
    for name,wanted in closure['files'].items():assert pin(ROOT/name)==wanted,name
    assert pin(REFERENCE)['sha256']=='3bad7d262b8809b1265c84c8e66d02ee38e7d4cff2d92014448976a9e161103c'
    assert pin(OLD/'complete.json')['sha256']=='f806908d2a4889398f330b43f8e5c00fa84eda39bb756f216adc66706775f0dc'
    assert pin(OLD/'replay/TranscribeReplay.dll')['sha256']=='335ca09d0e45e344068c484c92af9d0db43a6ae0accd1895d7ae7bb88b0afcf9'
    old=json.loads((OLD/'complete.json').read_text())
    assert old['application_passed'] and not old['full_numeric_gate'] and old['native_reproduced']
    prior=next(r for r in old['results'] if r['configuration']=='default')
    assert pin(OLD/'default.json')['sha256']==prior['result_sha256']
    assert pin(OLD/'default-audit.json')['sha256']==prior['audit_sha256']
    BASE.mkdir(exist_ok=False)
    files={str(p.relative_to(ROOT)):pin(p) for p in [receipt,REFERENCE,OLD/'complete.json',OLD/'default.json',OLD/'default-audit.json']}
    cores={}
    for role,source in [('baseline',PRODUCT),('candidate',CANDIDATE/'runtimes/candidate')]:
        target=BASE/'runtimes'/role;shutil.copytree(OLD/'replay',target)
        for path in source.glob('*.dll'):shutil.copy2(path,target/path.name)
        cores[role]=pin(target/'Lokad.Onnx.dll')
        assert pin(target/'Lokad.Onnx.Data.dll')['sha256']=='e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb'
        assert pin(target/'TranscribeReplay.dll')['sha256']=='335ca09d0e45e344068c484c92af9d0db43a6ae0accd1895d7ae7bb88b0afcf9'
        for path in target.iterdir():
            if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    assert cores['baseline']['sha256']=='d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4'
    assert cores['candidate']['sha256']=='469cb2d6a4558d917266434bd1f968c8b9f2762795b963800b2d945083852edd'
    manifest=json.loads(REFERENCE.read_text())
    for name,entry in manifest['files'].items():
        path=REFERENCE.parent/name;assert pin(path)=={k:entry[k] for k in ('bytes','sha256')};files[str(path.relative_to(ROOT))]=pin(path)
    for name,entry in manifest['assets']['files'].items():
        path=MODELS/name;assert pin(path)=={k:entry[k] for k in ('bytes','sha256')};files[str(path.relative_to(ROOT))]=pin(path)
    for path in TOOLS.iterdir():
        if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    source=ROOT/'tests/parakeet/transcribe/audit.py';files[str(source.relative_to(ROOT))]=pin(source)
    save(BASE/'manifest.json',dict(files=files,cores=cores,reference=str(REFERENCE.relative_to(ROOT)),models=str(MODELS.relative_to(ROOT)),
        jobs=['baseline','candidate'],limits=dict(seconds=1800,rss=8*1024**3,available=1024**3,preflight=10*1024**3,disk=20*1024**3,preflight_wait_seconds=3600),
        criterion='Application contracts; independently audit all native failures; no new failures, no worse existing failure maxima, and no baseline-relative change above scaled 1e-4',
        known_native_failures=['english-16k/step-26/outputs','english-frame-limit/step-26/outputs','english-repeat/step-26/outputs']))
    print(json.dumps(dict(files=len(files),manifest=pin(BASE/'manifest.json'),cores=cores)))


if __name__=='__main__':main()
