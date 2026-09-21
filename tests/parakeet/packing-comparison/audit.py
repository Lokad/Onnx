"""Retain every sample and apply the prospective identical-process controls."""
import hashlib
import importlib.util
import json
import statistics
from common import *
import numpy as np


def main():
    prepared=read(BASE/'prepared.json');assert prepared['passed'];verify(prepared['files'])
    for name,wanted in prepared['external_files'].items():assert pin(Path(name))==wanted,name
    state=read(BASE/'processes.json');assert state['complete'] and state['code']==0;terminal(state['supervisor'])
    assert [r['name'] for r in state['runs']]==[f'{i}-{role}' for i,role in enumerate(prepared['jobs'])]
    spec=importlib.util.spec_from_file_location('original_audio_auditor',ROOT/'tests/audio/comparison/audit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    manifest=read(INPUT);assert (manifest['warmup_passes'],manifest['measured_passes'],len(manifest['cases']))==(1,3,20)
    for case in manifest['cases']:
        pcm=np.load(ROOT/case['pcm']['path'],allow_pickle=False)
        assert pcm.dtype==np.float32 and pcm.shape==(case['samples'],) and np.isfinite(pcm).all()
        case['raw_sha256']=hashlib.sha256(pcm.tobytes()).hexdigest()
    workers=[];by_role={r:[] for r in ('production','512','2032','ort')};calls=0
    for index,(run,role) in enumerate(zip(state['runs'],prepared['jobs'],strict=True)):
        assert run['complete'] and run['code']==0 and run['application_passed'] and run['samples']>0
        for pid,birth in run['members'].items():terminal(dict(pid=int(pid),birth=birth))
        assert run['preflight']['available']>=14*1024**3
        folder=BASE/run['name'];result=read(folder/'output/result.json');module.validate_worker(result,manifest,'timing')
        assert result['manifest_sha256']==pin(INPUT)['sha256'];calls+=len(result['records'])
        if role=='ort':
            assert result['engine']=='ort' and result['onnxruntime']=='1.29.0' and result['numpy']=='2.2.4'
            assert result['native_binaries']==prepared['native_binaries'] and result['native_settings']==prepared['native_settings']
            assert result['runner_sha256']==pin(NATIVE)['sha256'] and result['adapter_sha256']==pin(NATIVE.with_name('native_adapters.py'))['sha256']
            assert result['python_binary_sha256']==prepared['external_files'][sys.executable]['sha256']
        else:
            assert result['engine']=='managed' and result['runtime']=='.NET 10.0.12' and result['processor_count']==1 and result['flags']=={}
            for key,name in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll'),('runner_sha256','AudioBenchmark.dll')]:assert result[key]==prepared['roles'][role][name]['sha256']
        assert [p.name for p in sorted((folder/'output').glob('[0-9][0-9][0-9].json'))]==[f'{i:03}.json' for i in range(80)]
        for ordinal,row in enumerate(result['records']):assert read(folder/'output'/f'{ordinal:03}.json')==row
        samples=[json.loads(line) for line in (BASE/'logs'/(run['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples)==run['samples'] and max(s['rss'] for s in samples)==run['peak_rss']
        assert all(s['seconds']<1800 and s['rss']<12*1024**3 and s['available']>=1024**3 and s['disk']>=20*1024**3 and s['output_bytes']<=1024**3
            and len(s['members'])<=1 and all(p['affinity']==[2] for p in s['members']) for s in samples)
        measured=[r for r in result['records'] if r['phase']=='measured'];assert len(measured)==60
        means={c['name']:statistics.fmean(r['seconds'] for r in measured if r['name']==c['name']) for c in manifest['cases']}
        observation=dict(index=index,role=role,corpus_seconds=sum(means.values()),clip_means=means,samples=run['samples'],peak_rss=run['peak_rss'])
        workers.append(observation);by_role[role].append(observation)
    assert calls==640 and all(len(v)==2 for v in by_role.values())
    controls={};means={}
    for role,rows in by_role.items():
        corpus=[r['corpus_seconds'] for r in rows];clip_ratios={c['name']:max(r['clip_means'][c['name']] for r in rows)/min(r['clip_means'][c['name']] for r in rows) for c in manifest['cases']}
        ratio=max(corpus)/min(corpus)
        controls[role]=dict(corpus_max_min=ratio,clip_max_min=clip_ratios,
            passed=ratio<=prepared['controls']['corpus_max_ratio'] and max(clip_ratios.values())<=prepared['controls']['clip_max_ratio'])
        means[role]=dict(corpus_seconds=statistics.fmean(corpus),clips={c['name']:statistics.fmean(r['clip_means'][c['name']] for r in rows) for c in manifest['cases']})
    attribution_valid=all(c['passed'] for c in controls.values());candidates={}
    for role in ('512','2032'):
        corpus=means[role]['corpus_seconds']/means['production']['corpus_seconds']
        ratios={name:value/means['production']['clips'][name] for name,value in means[role]['clips'].items()}
        candidates[role]=dict(corpus_ratio_to_production=corpus,clip_ratios_to_production=ratios,
            corpus_ratio_to_ort=means[role]['corpus_seconds']/means['ort']['corpus_seconds'],
            qualifies_for_amd=attribution_valid and corpus<=prepared['admission']['corpus_ratio'] and max(ratios.values())<=prepared['admission']['maximum_clip_ratio'])
    eligible=[r for r,c in candidates.items() if c['qualifies_for_amd']]
    selected=None
    if eligible:
        selected=min(eligible,key=lambda r:means[r]['corpus_seconds'])
        if len(eligible)==2 and means['512']['corpus_seconds']<=prepared['admission']['lower_budget_preference_ratio']*means['2032']['corpus_seconds']:selected='512'
    analysis=dict(passed=True,attribution_valid=attribution_valid,selected_for_amd=selected,controls=controls,candidates=candidates,means=means,workers=workers,
        calls=calls,warmup_calls=160,measured_calls=480,resources_samples=sum(r['samples'] for r in state['runs']),
        scope='Descriptive local full application trial; all variation retained; no calibrated parity or production promotion')
    save(BASE/'analysis.json',analysis)
    files=dict(prepared['files'])
    for p in BASE.rglob('*'):
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in TOOLS.iterdir():
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    assert not (BASE/'closed.json').exists();save(BASE/'closed.json',dict(passed=True,attribution_valid=attribution_valid,files=files,analysis=pin(BASE/'analysis.json')))
    print(json.dumps({k:v for k,v in analysis.items() if k not in ('workers','controls','means','candidates')}))
    print(json.dumps(dict(corpus_means={r:v['corpus_seconds'] for r,v in means.items()},candidates=candidates,closed=pin(BASE/'closed.json'))))


if __name__=='__main__':main()
