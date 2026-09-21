"""Freeze six complete-application workers after corrected-arithmetic qualification."""
import json
import shutil
from common import *


def main():
    prerequisites = [
        ('artifacts/parakeet-reduction-dispatch-20260921/closed.json', '0f5700924712ad313a840e28f704d22bb3f1c97c948264d5abd5769f31dfcaf1'),
        ('artifacts/parakeet-reduction-shared-v2-20260921/closed.json', '85c1137baf975881e36b30ba72298ee343f26f22af0062b4a6ef7e6de1341d0a'),
        ('artifacts/parakeet-reduction-dispatch-suites-20260921/closed.json', '1bdfbdaf664ebe1f54990b3a91c18e3cadef2f2d2f509c082dece8367b1d40de'),
        ('artifacts/parakeet-reduction-pyannote-20260921/closed.json', '0c9e134c4d7a6623373f7dffb2ac73b34cd225bcfa5ddd37d74a4d923b2b0f34')]
    for name, sha in prerequisites:
        receipt=ROOT/name;assert pin(receipt)['sha256']==sha
        closure=read(receipt);assert closure.get('passed', closure.get('native_numeric_passed', False))
        verify(closure['files'])
        identities=closure.get('terminal_identities',[])+closure.get('worker_identities',[])
        if 'supervisor' in closure:identities.append(closure['supervisor'])
        for identity in identities:terminal(identity)
    previous=read(OLD/'timing/01-parakeet-ort/result.json')
    assert previous['runner_sha256']==pin(NATIVE)['sha256'] and previous['adapter_sha256']==pin(NATIVE.with_name('native_adapters.py'))['sha256']
    assert previous['manifest_sha256']==pin(INPUT)['sha256']
    manifest=read(INPUT);assert (manifest['warmup_passes'],manifest['measured_passes'],len(manifest['cases']))==(1,3,20)
    BASE.mkdir();(BASE/'logs').mkdir();files={};roles={}
    for role in ('production','candidate'):
        target=BASE/('runtime-'+role);shutil.copytree(OLD/'bin',target)
        for p in PRODUCTION.glob('*.dll'):shutil.copy2(p,target/p.name)
        if role=='candidate':shutil.copy2(QUALIFICATION/'runtime/Lokad.Onnx.dll',target/'Lokad.Onnx.dll')
        assert pin(target/'AudioBenchmark.dll')['sha256']=='7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
        roles[role]={n:pin(target/n) for n in ('Lokad.Onnx.dll','Lokad.Onnx.Data.dll','AudioBenchmark.dll')}
        for p in target.iterdir():
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    assert roles['production']['Lokad.Onnx.dll']['sha256']=='d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4'
    assert roles['candidate']['Lokad.Onnx.dll']['sha256']=='f2c292cb6856e7ec80769e983df1f01512e98ad8d4d6d6ec3d424c6028e8f791'
    assert roles['production']['Lokad.Onnx.Data.dll']==roles['candidate']['Lokad.Onnx.Data.dll']==pin(QUALIFICATION/'runtime/Lokad.Onnx.Data.dll')
    for p in (BASE/'runtime-production').iterdir():
        if p.is_file() and p.name!='Lokad.Onnx.dll':assert pin(p)==pin(BASE/'runtime-candidate'/p.name)
    for value in [*manifest['models'].values(),manifest['reference'],manifest['upstream'],*[c['pcm'] for c in manifest['cases']]]:
        assert pin(ROOT/value['path'])=={k:value[k] for k in ('bytes','sha256')};files[value['path']]=pin(ROOT/value['path'])
    external={}
    for name,sha in previous['native_binaries'].items():
        p=Path(name);assert pin(p)['sha256']==sha;external[name]=pin(p)
    external[sys.executable]=pin(Path(sys.executable))
    for p in [*[ROOT/name for name,sha in prerequisites],INPUT,NATIVE,NATIVE.with_name('native_adapters.py'),MONITOR_PATH,OLD/'timing/01-parakeet-ort/result.json',ROOT/'tests/audio/comparison/audit.py',*TOOLS.iterdir()]:
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    save(BASE/'prepared.json',dict(passed=True,files=files,external_files=external,roles=roles,
        native_binaries=previous['native_binaries'],native_settings=previous['native_settings'],
        jobs=['production','candidate','ort','ort','candidate','production'],
        limits=dict(preflight_gib=14,rss_gib=12,seconds=1800),controls=dict(corpus_max_ratio=1.10,clip_max_ratio=1.20),
        admission=dict(corpus_ratio=.95,maximum_clip_ratio=1.05),
        scope='Local descriptive whole-application trial; fixed controls; no calibrated parity or AMD/promotion claim'))
    print(json.dumps(dict(passed=True,prepared=pin(BASE/'prepared.json'),jobs=6,requests=480)))


if __name__=='__main__':main()
