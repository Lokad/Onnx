"""Freeze eight fresh whole-application workers after higher-budget qualification."""
import json
import shutil
from common import *


def main():
    closure=read(QUALIFICATION/'closed.json');assert closure['passed'] and closure['regression_passed'];verify(closure['files'])
    state=read(QUALIFICATION/'processes.json');assert state['complete'] and state['code']==0;terminal(state['supervisor'])
    for row in state['runs']:
        for pid,birth in row['members'].items():terminal(dict(pid=int(pid),birth=birth))
    previous=read(OLD/'timing/01-parakeet-ort/result.json')
    assert previous['runner_sha256']==pin(NATIVE)['sha256'] and previous['adapter_sha256']==pin(NATIVE.with_name('native_adapters.py'))['sha256']
    assert previous['manifest_sha256']==pin(INPUT)['sha256']
    manifest=read(INPUT);assert (manifest['warmup_passes'],manifest['measured_passes'],len(manifest['cases']))==(1,3,20)
    BASE.mkdir();(BASE/'logs').mkdir();files={};roles={}
    for role,source in [('production',PRODUCTION),('512',QUALIFICATION/'runtime-512'),('2032',QUALIFICATION/'runtime-2032')]:
        target=BASE/('runtime-'+role);shutil.copytree(OLD/'bin',target)
        for p in source.glob('*.dll'):shutil.copy2(p,target/p.name)
        assert pin(target/'AudioBenchmark.dll')['sha256']=='7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
        roles[role]={n:pin(target/n) for n in ('Lokad.Onnx.dll','Lokad.Onnx.Data.dll','AudioBenchmark.dll')}
        for p in target.iterdir():
            if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    assert roles['production']['Lokad.Onnx.dll']['sha256']=='d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4'
    for value in [*manifest['models'].values(),manifest['reference'],manifest['upstream'],*[c['pcm'] for c in manifest['cases']]]:
        assert pin(ROOT/value['path'])=={k:value[k] for k in ('bytes','sha256')};files[value['path']]=pin(ROOT/value['path'])
    external={}
    for name,sha in previous['native_binaries'].items():
        p=Path(name);assert pin(p)['sha256']==sha;external[name]=pin(p)
    external[sys.executable]=pin(Path(sys.executable))
    for p in [QUALIFICATION/'closed.json',INPUT,NATIVE,NATIVE.with_name('native_adapters.py'),MONITOR_PATH,OLD/'timing/01-parakeet-ort/result.json',ROOT/'tests/audio/comparison/audit.py',*TOOLS.iterdir()]:
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    save(BASE/'prepared.json',dict(passed=True,files=files,external_files=external,roles=roles,
        native_binaries=previous['native_binaries'],native_settings=previous['native_settings'],
        jobs=['production','512','2032','ort','ort','2032','512','production'],
        limits=dict(preflight_gib=14,rss_gib=12,seconds=1800),controls=dict(corpus_max_ratio=1.10,clip_max_ratio=1.20),
        admission=dict(corpus_ratio=.95,maximum_clip_ratio=1.05,lower_budget_preference_ratio=1.02),
        scope='Local descriptive whole-application trial; fixed controls; no calibrated parity or AMD/promotion claim'))
    print(json.dumps(dict(passed=True,prepared=pin(BASE/'prepared.json'),jobs=8,requests=640)))


if __name__=='__main__':main()
