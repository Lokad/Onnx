"""Audit complete application evidence and retain all multilingual recognition errors."""
from pathlib import Path
import argparse
import importlib.metadata
import json
import math
import psutil
from common import CORE,DATA,LOCALES,pin,read,sha,write_new
from scoring import metrics,total


def validate_decision(family,value):
    assert isinstance(value,dict) and type(value['text']) is str
    tokens=value['token_ids'];assert isinstance(tokens,list) and all(type(t) is int for t in tokens)
    if family=='parakeet':
        assert set(value)=={'text','token_ids','frame_indices','duration_frames','stop_reason','encoded_frames','decoder_calls'}
        assert len(tokens)<=4096 and all(0<=t<8192 for t in tokens)
        assert type(value['encoded_frames']) is int and value['encoded_frames']>0
        assert type(value['decoder_calls']) is int and value['decoder_calls']>=len(tokens)
        assert len(value['frame_indices'])==len(value['duration_frames'])==len(tokens)
        assert all(type(f) is int and 0<=f<value['encoded_frames'] for f in value['frame_indices'])
        assert value['frame_indices']==sorted(value['frame_indices'])
        assert all(type(d) is int and 0<=d<=4 for d in value['duration_frames'])
        assert value['stop_reason'] in ('EndOfAudio','TokenLimit')
        if value['stop_reason']=='TokenLimit':assert len(tokens)==4096
    else:
        assert family=='whisper' and set(value)=={'text','token_ids','stop_reason','skipped_as_no_speech'}
        assert 0<len(tokens)<=444 and all(0<=t<=50257 for t in tokens)
        assert type(value['skipped_as_no_speech']) is bool
        assert value['stop_reason'] in ('EndToken','TokenLimit')
        if value['stop_reason']=='EndToken':assert tokens[-1]==50257
        else:assert len(tokens)==444 and tokens[-1]!=50257
        if value['skipped_as_no_speech']:assert value['text']==''


def validate_records(family,engine,rows,cases):
    assert len(cases)==40 and len(rows)==41
    for index,(row,case) in enumerate(zip(rows,cases+cases[:1],strict=True)):
        assert (row['name'],row['language'],row['pcm_sha256'],row['repeat'])==(case['name'],case['language'],case['pcm_sha256'],index==40)
        assert row['input_and_held_results_unchanged'] is True
        assert all(type(row[k]) is int for k in ('start_ticks','end_ticks','frequency'))
        assert 0<=row['start_ticks']<row['end_ticks'] and row['frequency']>0
        assert math.isfinite(row['seconds']) and 0<row['seconds']<3600
        assert row['seconds']==(row['end_ticks']-row['start_ticks'])/row['frequency']
        validate_decision(family,row['decision'])
        if engine=='managed':
            fields=dict(text='Text',token_ids='TokenIds',stop_reason='StopReason')
            fields.update(dict(frame_indices='FrameIndices',duration_frames='DurationFrames',encoded_frames='EncodedFrames',decoder_calls='DecoderCalls')
                          if family=='parakeet' else dict(skipped_as_no_speech='SkippedAsNoSpeech'))
            assert row['decision']=={name:row['result'][prop] for name,prop in fields.items()}
    assert rows[0]['decision']==rows[-1]['decision']
    if engine=='managed':assert rows[0]['result']==rows[-1]['result']


def audit(base):
    root=Path(__file__).resolve().parents[3]
    frozen=read(base/'frozen.json');identity=read(base/'run/identity.json')
    assert identity['complete'] is True and 'error' not in identity and len(identity['runs'])==4
    assert identity['frozen_sha256']==sha(base/'frozen.json')
    assert identity['limits']==dict(rss=20*1024**3,seconds=3600,available=1024**3)
    for name,wanted in frozen['files'].items():assert pin(root/name)==wanted,name
    for name,version in {'jiwer':'4.0.0','rapidfuzz':'3.14.6','psutil':'7.0.0'}.items():assert importlib.metadata.version(name)==version,name
    audio=read(base/'inputs/audio.json');input_audit=read(base/'input-audit.json')
    assert input_audit['passed'] and input_audit['audio_sha256']==sha(base/'inputs/audio.json')
    for name,wanted in input_audit['files'].items():assert pin(base/'inputs'/name)==wanted,name
    results={};resources=[];births={(identity['supervisor']['pid'],identity['supervisor']['birth'])}
    order=[('parakeet','ort'),('parakeet','managed'),('whisper','ort'),('whisper','managed')]
    for number,(row,(family,engine)) in enumerate(zip(identity['runs'],order,strict=True)):
        assert (row['name'],row['family'],row['engine'])==(f'{number:02d}-{family}-{engine}',family,engine)
        assert row['code']==0 and 0<row['seconds']<3600
        if engine=='managed':assert row['preflight_available']>=20*1024**3
        samples=[json.loads(line) for line in (base/'run'/(row['name']+'-samples.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples'] and len(samples)>1
        seen={};last=-1;peak=0;available=[]
        for sample in samples:
            assert last<=sample['seconds']<row['seconds'];last=sample['seconds']
            assert sample['available']>=1024**3;available.append(sample['available'])
            assert len({m['pid'] for m in sample['members']})==len(sample['members'])
            rss=sum(m['rss'] for m in sample['members']);assert 0<=rss<20*1024**3;peak=max(peak,rss)
            for member in sample['members']:
                assert member['affinity']==[2] and member['rss']>=0 and member['cpu_seconds']>=0
                assert seen.get(str(member['pid']),member['birth'])==member['birth']
                seen[str(member['pid'])]=member['birth'];births.add((member['pid'],member['birth']))
        assert seen==row['members'] and seen[str(row['pid'])]==row['birth'] and peak==row['peak_rss']
        assert row['accounting']['valid'] and math.isfinite(row['accounting']['foreign_cpu_fraction'])
        manifest_path=base/'manifests'/(family+'.json');manifest=read(manifest_path)
        result=read(base/'run'/row['name']/'result.json')
        assert result['schema']==1 and result['protocol']=='multilingual-noise-asr-v2' and result['passed'] is True
        assert (result['family'],result['engine'])==(family,engine) and result['affinity']==4 and not result['flags']
        assert result['manifest_sha256']==sha(manifest_path) and result['audio_sha256']==sha(base/'inputs/audio.json')
        if engine=='managed':
            assert result['core_sha256']==CORE and result['data_sha256']==DATA and result['runtime']=='.NET 10.0.12'
            assert result['runner_sha256']==sha(base/'bin/MultilingualReplay.dll')
        else:
            assert result['numpy']=='2.2.4' and result['onnxruntime']=='1.29.0'
            assert result['ort_binary'] in [{k:v[k] for k in ('bytes','sha256')} for v in manifest['native_binaries'].values()]
        validate_records(family,engine,result['cases'],audio['cases'])
        assert {p.name for p in (base/'run'/row['name']).iterdir()}=={'result.json'}|{f'{i:02d}.json' for i in range(41)}
        for i,case in enumerate(result['cases']):assert read(base/'run'/row['name']/f'{i:02d}.json')==case
        results[(family,engine)]=result['cases']
        resources.append(dict(name=row['name'],peak_rss=peak,minimum_available=min(available),samples=len(samples),seconds=row['seconds'],foreign_cpu_fraction=row['accounting']['foreign_cpu_fraction']))
    for pid,birth in births:assert not psutil.pid_exists(pid) or psutil.Process(pid).create_time()!=birth,(pid,birth)
    models={}
    for family in ('parakeet','whisper'):
        native,managed=results[(family,'ort')],results[(family,'managed')]
        rows=[]
        for case,left,right in zip(audio['cases'],native[:40],managed[:40],strict=True):
            rows.append(dict(name=case['name'],locale=case['locale'],language=case['language'],condition=case['condition'],
                reference_text=case['reference_text'],native=left['decision'],managed=right['decision'],application_matches=left['decision']==right['decision'],
                native_metrics=metrics(case['reference_text'],left['decision']['text']),managed_metrics=metrics(case['reference_text'],right['decision']['text']),
                native_seconds=left['seconds'],managed_seconds=right['seconds']))
        groups=[]
        for locale in [x[0] for x in LOCALES]+['all']:
            for condition in ('clean','noise10db'):
                subset=[r for r in rows if (locale=='all' or r['locale']==locale) and r['condition']==condition]
                assert len(subset)==(20 if locale=='all' else 4)
                groups.append(dict(locale=locale,condition=condition,recordings=len(subset),native=total([r['native_metrics'] for r in subset]),managed=total([r['managed_metrics'] for r in subset])))
        models[family]=dict(application_passed=all(a['decision']==b['decision'] for a,b in zip(native,managed,strict=True)),cases=rows,groups=groups)
    return dict(schema=1,execution_passed=True,application_passed=all(m['application_passed'] for m in models.values()),
        requests=164,recordings=20,cases=40,audio_seconds=input_audit['audio_seconds'],models=models,resources=resources,
        terminal_processes=[dict(pid=pid,birth=birth) for pid,birth in sorted(births)],
        frozen_sha256=sha(base/'frozen.json'),identity_sha256=sha(base/'run/identity.json'),auditor_sha256=sha(Path(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists();result=audit(args.artifact.resolve());write_new(args.output,result)
    print('Complete evidence audit:',result['requests'],'requests; native application agreement:',result['application_passed'])
