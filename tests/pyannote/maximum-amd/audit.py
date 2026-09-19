"""Check complete maximum-duration diarization application and AMD resources."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import numpy as np
from prepare import sha, read, pin, write_new


def compare(actual,expected,duration,windows):
    assert type(actual['Status']) is int and actual['Status']==0 and expected['status']=='Completed'
    assert actual['AudioDuration']==duration==expected['seconds'] and type(actual['Windows']) is int and actual['Windows']==windows
    assert len(expected['windows'])==windows
    assert len(actual['Speakers'])==len(expected['speakers'])>0
    maximum=0.
    for index,(a,b) in enumerate(zip(actual['Speakers'],expected['speakers'],strict=True)):
        assert type(a['Speaker']) is int and a['Speaker']==b['speaker']==index
        assert type(a['HasEmbedding']) is bool and a['HasEmbedding']==b['has_embedding']
        x,y=np.asarray(a['Centroid']),np.asarray(b['centroid'])
        assert x.shape==y.shape==(256,) and np.isfinite(x).all() and np.isfinite(y).all()
        if not a['HasEmbedding']:assert np.all(x==0)
        maximum=max(maximum,float(np.max(np.abs(x-y)/np.maximum(1,np.abs(y)))))
    assert maximum<=1e-4,'Centroid numerical gate failed'
    for field,key in [('Intervals','intervals'),('ExclusiveIntervals','exclusive_intervals')]:
        assert len(actual[field])==len(expected[key])
        for a,(start,end,speaker) in zip(actual[field],expected[key],strict=True):
            assert math.isfinite(a['Start']) and math.isfinite(a['End']) and 0<=a['Start']<a['End']<=duration
            assert type(a['Speaker']) is int and 0<=a['Speaker']<len(actual['Speakers']) and a['Speaker']==speaker
            assert abs(a['Start']-start)<=1e-12 and abs(a['End']-end)<=1e-12,'Timeline boundary differs'
    ordered=sorted(actual['ExclusiveIntervals'],key=lambda item:item['Start'])
    assert all(a['End']<=b['Start'] for a,b in zip(ordered,ordered[1:]))
    return maximum


def application(managed,long,short,windows,input_sha):
    assert managed['passed'] is True and managed['held_unchanged'] is True and managed['input_unchanged'] is True
    assert managed['refusals']==['one-sample-over-limit','canceled']
    assert managed['samples']==9600000 and managed['copies']==20 and managed['input_sha256']==input_sha
    assert managed['empty']==dict(Intervals=[],ExclusiveIntervals=[],Speakers=[],Status=1,AudioDuration=0,Windows=0)
    for name in ('full_request_seconds','recovery_seconds'):
        assert math.isfinite(managed[name]) and managed[name]>0
    maximum=compare(managed['result'],long['cases'][0],600,591)
    recovery=compare(managed['recovery'],short['cases'][0],30,21)
    assert managed['recovery_exact_timeline'] is True and managed['recovery_centroid_error']==recovery
    # Windows is a different, older product build; compare discrete outputs and
    # keep centroid differences separately under the same existing gate.
    cross=0.
    for name in ('result','recovery'):
        a,b=managed[name],windows[name]
        for field in ('Status','AudioDuration','Windows','Intervals','ExclusiveIntervals'):
            assert a[field]==b[field], 'Retained Windows application differs'
        assert len(a['Speakers'])==len(b['Speakers'])
        for x,y in zip(a['Speakers'],b['Speakers'],strict=True):
            assert x['Speaker']==y['Speaker'] and x['HasEmbedding']==y['HasEmbedding']
            values,wanted=np.asarray(x['Centroid']),np.asarray(y['Centroid'])
            cross=max(cross,float(np.max(np.abs(values-wanted)/np.maximum(1,np.abs(wanted)))))
    assert cross<=1e-4
    return dict(request_seconds=managed['full_request_seconds'],recovery_seconds=managed['recovery_seconds'],
        windows=591,speakers=len(managed['result']['Speakers']),ordinary_intervals=len(managed['result']['Intervals']),
        exclusive_intervals=len(managed['result']['ExclusiveIntervals']),maximum_native_centroid_error=maximum,
        maximum_recovery_centroid_error=recovery,maximum_windows_centroid_difference=cross)


def resources(identity,samples,collection):
    assert identity['schema']==1 and identity['complete'] is True and 'error' not in identity
    assert identity['supervisor']['affinity']=='0'
    assert identity['limits']==dict(rss=8*1024**3,seconds=1800,available_memory=256*1024**2)
    assert len(identity['runs'])==1
    row=identity['runs'][0]
    assert row['name']=='managed' and row['code']==0 and 0<row['seconds']<1800
    assert identity['started']<=row['started']<=row['ended']<=identity['ended']
    assert len(samples)==row['samples'] and len(samples)>1
    previous,peak,minimum=-1,0,math.inf;births,cpu={},{}
    for sample in samples:
        assert previous<=sample['seconds']<row['seconds'];previous=sample['seconds']
        assert sample['available_memory']>=256*1024**2;minimum=min(minimum,sample['available_memory'])
        seen=set()
        for member in sample['members']:
            pid,start=member['pid'],member['start']
            assert member['group']==row['pid'] and member['affinity']=='2' and member['state']!='Z'
            assert start>=row['start'] and births.get(str(pid),start)==start
            assert member['rss']>=0 and member['cpu_seconds']>=cpu.get(pid,0) and pid not in seen
            if pid==row['pid']:assert start==row['start']
            births[str(pid)]=start;cpu[pid]=member['cpu_seconds'];seen.add(pid)
        rss=sum(m['rss'] for m in sample['members']);assert rss<8*1024**3;peak=max(peak,rss)
    assert births==row['members'] and births[str(row['pid'])]==row['start'] and peak==row['peak_rss']
    expected={(int(pid),start) for pid,start in births.items()}|{(identity['supervisor']['pid'],identity['supervisor']['start'])}
    assert expected=={(p['pid'],p['start']) for p in collection['terminal_processes']}
    assert collection['complete'] is True and collection['code']==0 and collection['checkout']=='172181fc5ab4eb2bdc2eb7f37e80d25e482a0887'
    return dict(seconds=row['seconds'],samples=len(samples),peak_rss=peak,minimum_available_memory=minimum,
        terminal_processes=[dict(pid=pid,start=start) for pid,start in sorted(expected)])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    base=args.artifact.resolve();payload,collected=base/'payload',base/'collected'
    bundle,collection=read(payload/'bundle.json'),read(collected/'collection.json')
    assert sha(payload/'bundle.json')==sha(collected/'bundle.json')==read(base/'preparation.json')['bundle_sha256']
    assert sha(collected/'collection.json')==read(base/'download.json')['collection_sha256']
    assert {p.relative_to(collected).as_posix() for p in collected.rglob('*') if p.is_file()}==set(collection['files'])|{'collection.json'}
    for name,expected in collection['files'].items():assert pin(collected/name)==expected,name
    for name,expected in bundle['files'].items():
        assert pin(payload/name)==expected,name
        if name in collection['files']:assert pin(collected/name)==expected
    receipt=read(payload/'reference/long-receipt.json')
    assert sha(payload/'reference/long-receipt.json')==bundle['original_long_receipt_sha256']=='e8c8371f887ea0b5a194d9e370dc33c164919c6362af5a0c82358a2f1bce314a'
    for name,original in [('long-native.json','native-reference/manifest.json'),('windows.json','default/result.json'),('resource-receipt.json','default/receipt.json')]:
        assert sha(payload/'reference'/name)==receipt['files'][original]
    long,short=read(payload/'reference/long-native.json'),read(payload/'reference/manifest.json')
    assert len(long['cases'])==1 and long['cases'][0]['name']=='repeat-dialogue-600s' and len(long['files'])==bundle['native_arrays_verified']==4146
    assert receipt['native_tie_differences']==dict(ordinary=0,exclusive=98)
    assert {key:item['sha256'] for key,item in bundle['models'].items()}==long['models']==short['models']
    resource=read(payload/'reference/resource-receipt.json')
    managed=read(collected/'result/managed.json')
    assert managed['flags']=={} and managed['runtime']=='.NET 10.0.8' and managed['avx2'] is True and managed['avx512'] is True
    assert managed['reference_sha256']==sha(payload/'reference/manifest.json')
    for name in ('Lokad.Onnx','Lokad.Onnx.Data','LimitProbe'):
        assert managed['assemblies'][name]['sha256']==sha(payload/'bin'/(name+'.dll'))
    assert managed['assemblies']['Lokad.Onnx']['sha256']=='05884cfd524cc7130321f5dc1bcd0af17dddc7b97e8428d2d2f59e00edb795c2'
    assert managed['assemblies']['Lokad.Onnx.Data']['sha256']=='27598aa8d8c6b97a1415302cf3aaced1adcf53b20b64734c0c0e047492ca069d'
    pcm_path=payload/'reference'/short['cases'][0]['pcm'];assert sha(pcm_path)==short['files'][pcm_path.name]['sha256']
    pcm=np.load(pcm_path,allow_pickle=False);assert pcm.dtype==np.float32 and pcm.shape==(480000,)
    assert hashlib.sha256(np.tile(pcm,20).astype('<f4',copy=False).tobytes()).hexdigest()==bundle['input_sha256']==receipt['input_sha256']==resource['input_sha256']
    result=application(managed,long,short,read(payload/'reference/windows.json'),bundle['input_sha256'])
    identity=read(collected/'result/identity.json');assert identity['bundle_sha256']==sha(payload/'bundle.json')
    samples=[json.loads(line) for line in (collected/'result/managed-samples.jsonl').read_text(encoding='utf-8').splitlines()]
    limits=resources(identity,samples,collection)
    assert result['request_seconds']+result['recovery_seconds']<limits['seconds']
    assert 0<managed['peak']<8*1024**3
    result.update(schema=1,passed=True,resources=limits,process_peak_working_set=managed['peak'],
        bundle_sha256=sha(payload/'bundle.json'),collection_sha256=sha(collected/'collection.json'),
        managed_sha256=sha(collected/'result/managed.json'),auditor_sha256=sha(Path(__file__)),
        product_source=bundle['product_source'],previous_windows_product=resource['product_source'],
        native_tie_differences=receipt['native_tie_differences'],full_managed_tensor_gate='not measured',
        scope='One synthetic600-second AMD public API request with native/Windows public comparison and short recovery; no independent natural accuracy or calibrated timing claim')
    write_new(args.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
