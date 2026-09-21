"""Freeze a fresh complete-application Windows comparison of candidate and ORT."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-optimized-ort-20260921'
OLD=ROOT/'artifacts/audio-ort-baseline-v2-20260919'
INPUT=OLD/'inputs/pyannote.json'
CANDIDATE=ROOT/'artifacts/pyannote-lstm-output-lanes-20260921'
NATIVE=ROOT/'tests/audio/comparison/native.py'


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')


def clean_env():
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_','omp_','mkl_','openblas_','blis_','numexpr_'))}
    env.update(PYTHONUTF8='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(str(ROOT/path) for path in
        ['artifacts/pyannote-diarization-20260919/python','artifacts/pyannote-clustering-20260919/python']))
    env.update({k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
    return env


def main():
    receipt=CANDIDATE/'qualification-closed.json';closed=json.loads(receipt.read_text());assert closed['passed']
    assert pin(receipt)['sha256']=='cf950ec5cedf702c1af38decc377cd516a5d8b652f77853d03d2c81db0b5bf53'
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    prior=json.loads((OLD/'receipt.json').read_text());assert prior['complete'] and prior['all_owned_processes_terminal']
    assert pin(OLD/'receipt.json')['sha256']=='b34f854d9e1c2fa5efeab46a628a8d632d1fb27f865652707281116800b8341a'
    previous=json.loads((OLD/'timing/05-pyannote-ort/result.json').read_text())
    assert previous['manifest_sha256']==pin(INPUT)['sha256']
    assert previous['runner_sha256']==pin(NATIVE)['sha256']
    assert previous['adapter_sha256']==pin(NATIVE.with_name('native_adapters.py'))['sha256']
    assert pin(OLD/'bin/AudioBenchmark.dll')['sha256']=='7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
    BASE.mkdir(exist_ok=False);shutil.copytree(OLD/'bin',BASE/'bin')
    for path in (CANDIDATE/'runtimes/candidate').glob('*.dll'):shutil.copy2(path,BASE/'bin'/path.name)
    assert pin(BASE/'bin/Lokad.Onnx.dll')['sha256']=='469cb2d6a4558d917266434bd1f968c8b9f2762795b963800b2d945083852edd'
    assert pin(BASE/'bin/Lokad.Onnx.Data.dll')['sha256']=='e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb'
    assert pin(BASE/'bin/AudioBenchmark.dll')['sha256']=='7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
    paths=[receipt,OLD/'receipt.json',OLD/'timing/05-pyannote-ort/result.json',INPUT,NATIVE,NATIVE.with_name('native_adapters.py'),
        ROOT/'tests/pyannote/diarization/native_rules.py',Path(sys.executable)]
    paths += [ROOT/'tests/pyannote/natural-meetings'/name for name in ['audit.py','common.py']]
    manifest=json.loads(INPUT.read_text());assert (manifest['warmup_passes'],manifest['measured_passes'])==(1,3) and len(manifest['cases'])==4
    assets=list(manifest['models'].values())+list(manifest['native_assets'].values())+list(manifest['upstream'].values())+[manifest['reference']]+[c['pcm'] for c in manifest['cases']]
    for item in assets:
        path=ROOT/item['path'];assert pin(path)=={k:item[k] for k in ['bytes','sha256']};paths.append(path)
    for path,sha in previous['native_binaries'].items():
        path=Path(path);assert pin(path)['sha256']==sha;paths.append(path)
    paths+=list((BASE/'bin').glob('*'))+list(TOOLS.glob('*'))
    files={str(p.resolve()):pin(p) for p in paths if p.is_file()}
    plan=dict(jobs=['candidate','ort','ort','candidate'],input=str(INPUT),files=files,
        core=pin(BASE/'bin/Lokad.Onnx.dll'),data=pin(BASE/'bin/Lokad.Onnx.Data.dll'),consumer=pin(BASE/'bin/AudioBenchmark.dll'),
        native_binaries=previous['native_binaries'],native_settings=previous['native_settings'],
        limits=dict(seconds=1800,rss=8*1024**3,available=1024**3,preflight=10*1024**3,disk=20*1024**3,preflight_wait_seconds=3600),
        protocol='Two fresh processes per engine, candidate/ORT/ORT/candidate; each one full warmup and three measured passes; all calls retained; descriptive local comparison, no calibrated parity or AMD inference')
    save(BASE/'prepared.json',plan);print(json.dumps(dict(files=len(files),prepared=pin(BASE/'prepared.json'),protocol=plan['protocol'])))


if __name__=='__main__':main()
