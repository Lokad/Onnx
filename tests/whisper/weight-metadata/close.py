"""Audit the full controlled snapshot transitions on both hosts and close their evidence."""
from pathlib import Path
import copy,json,shutil,sys
from run import ROOT,BASE,PRODUCT,LIMITS,pin,read,write
sys.path.insert(0,str(ROOT/'tests/whisper/weight-sharing'))
from deploy import ssh
import psutil

RENAMES={'first':('folded:Transpose_1010','model.decoder.embed_tokens.weight_transposed'),
         'past':('folded:Transpose_801','model.decoder.embed_tokens.weight_transposed')}


def main():
    census=read(PRODUCT/'weight-census.json');hosts=[];original_checks=0;output_pairs=0;births=[]
    for host,base in [('windows',BASE),('amd',BASE/'amd-v2/collected')]:
        state=read(base/'identity.json');frozen=read(base/'frozen.json');assert state['complete'] and state['code']==0
        assert state['frozen']==pin(base/'frozen.json') and frozen['limits']==LIMITS
        for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
        if host=='windows':
            for name,wanted in frozen['external'].items():assert pin(Path(name))==wanted,name
        else:
            receipt=read(base/'collection.json');assert receipt['code']==0
            for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
        own_births=[state['supervisor']]+[r['child'] for r in state['runs']]
        if host=='windows':
            for b in own_births:
                try:assert psutil.Process(b['pid']).create_time()!=b['birth']
                except psutil.NoSuchProcess:pass
        else:
            script='''import sys,json
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
for b in %r:
 try:assert psutil.Process(b['pid']).create_time()!=b['birth']
 except psutil.NoSuchProcess:pass
print('terminal')
'''%own_births
            assert ssh(script).strip()=='terminal'
        births.append(dict(host=host,births=own_births));workers=[];values=[]
        for run in state['runs']:
            folder=base/run['mode'];assert run['complete'] and run['code']==0 and 'error' not in run
            assert pin(folder/'worker/result.json')==run['result'];v=read(folder/'worker/result.json');values.append(v)
            assert v['passed'] and v['held_outputs_unchanged'] and v['input_unchanged'] and v['flags']=={}
            assert v['runtime']==('.NET 10.0.12' if host=='windows' else '.NET 10.0.8') and v['affinity']==4 and v['processor_count']==1
            for key,name in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll'),('runner_sha256','WhisperWeightMetadata.dll')]:assert v[key]==pin(BASE/'bin'/name)['sha256']
            assert v['logical_shared_bytes']==(635187200 if run['mode']=='shared' else 0)
            assert len(v['snapshots'])==5 and len(v['calls'])==4
            before=v['snapshots'][0];changes=[]
            for index,snapshot in enumerate(v['snapshots']):
                assert read(folder/'worker'/f'snapshot-{index:02}.json')==snapshot
                expected=copy.deepcopy(before);expected['stage']=snapshot['stage']
                for g,(key,name) in RENAMES.items():
                    if index>=(1 if g=='first' else 2):
                        match=next(r for r in expected[g]['initializers'] if r['name']==key)
                        assert match['tensor_name']==key;match['tensor_name']=name
                        node=next(n for n in before[g]['nodes'] if n['Name']==key.removeprefix('folded:'))
                        assert node['op']=='Transpose' and node['Outputs']==[name]
                        changes.append(dict(stage=snapshot['stage'],graph=g,name=key,field='tensor_name',before=key,after=name))
                assert snapshot==expected,(host,run['mode'],index)
                for g,original in zip(['first','past'],census['graphs'],strict=True):
                    entries={r['name']:r for r in snapshot[g]['initializers']}
                    for old in original['rows']:
                        actual=entries[old['name']]
                        assert actual['shape']==old['shape'] and actual['bytes']==old['bytes'] and actual['sha256']==old['sha256'] and actual['type']=={1:'Float',7:'Int64'}[old['type']]
                        original_checks+=1
            for index,call in enumerate(v['calls']):assert read(folder/'worker'/f'call-{index:02}.json')==call
            for i in range(2):assert v['calls'][i]['outputs']==v['calls'][i+2]['outputs'];output_pairs+=len(v['calls'][i]['outputs'])
            samples=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
            assert len(samples)==run['samples'] and 0<run['seconds']<LIMITS['seconds']
            assert run['preflight_available']>=LIMITS['preflight'] and run['preflight_disk']>=LIMITS['preflight_disk']
            previous=0
            for sample in samples:
                assert previous<sample['seconds']<run['seconds'];previous=sample['seconds']
                assert sample['rss']<LIMITS['rss'] and sample['available']>=LIMITS['available'] and sample['disk']>=LIMITS['disk'] and sample['affinity']==[2]
            assert run['peak_rss']==max(s['rss'] for s in samples)
            workers.append(dict(mode=run['mode'],changes=changes,calls=4,samples=len(samples),peak_rss=run['peak_rss'],min_available=min(s['available'] for s in samples),result=run['result']))
        for a,b in zip(values[0]['calls'],values[1]['calls'],strict=True):assert a==b;output_pairs+=len(a['outputs'])
        assert values[0]['snapshots'][0]['unique_payload_bytes']-values[1]['snapshots'][0]['unique_payload_bytes']==635187200
        hosts.append(dict(host=host,workers=workers))
    assert original_checks==11360
    summary=dict(passed=True,scope='Controlled decoder-only metadata diagnosis; original AMD sharing campaign remains failed',
        hosts=hosts,original_initializer_checks=original_checks,exact_output_pairs=output_pairs,calls=16,births=births)
    tracked=Path(__file__).parent;report=tracked/'results-20260920.md';data=tracked/'observations-20260920.json';write(data,summary)
    text=f'''# Decoder snapshot mismatch is reproduced by two cached tensor names

The controlled diagnostic on **Windows and AMD** finds exactly the same two
metadata transitions with and without weight sharing. No initializer byte hash,
shape, type, membership, graph binding, packing total or backing-storage count
changes. All {output_pairs} output-payload comparisons across repeats and the
same-host unshared/shared controls match exactly.

| Decoder | Initializer key | Tensor name after first execution |
|---|---|---|
| First | `folded:Transpose_1010` | `model.decoder.embed_tokens.weight_transposed` |
| Past | `folded:Transpose_801` | `model.decoder.embed_tokens.weight_transposed` |

Before execution each tensor's name equals its initializer key. The names change
once, when the corresponding graph first runs, and stay stable on the repeat.
`Node.TransposePrepared` returns the cached tensor itself; graph execution assigns
the node's output name when binding that result. This existing behavior occurs
in the unshared control as well. Names are metadata; payload bytes remain unchanged.

Four fresh processes execute sixteen graph calls in total. Each runs the actual
decoder pair twice with a zero hidden-state input and fixed token IDs, retaining
all actual outputs and original feeds. It saves all twenty full snapshots before
comparison. All {original_checks:,} checks against the original serialized
initializer census pass, including every original initializer at every stage.
All {sum(w['samples'] for h in hosts for w in h['workers'])} resource samples pass
120-second/8 GiB RSS/4 GiB available-memory bounds. Every original process identity
is terminal. No full speech inference or native ORT comparison occurs here.

This reproduces a specific flaw in the earlier snapshot-equality assumption.
The [original AMD campaign](../weight-sharing/failure-20260920.md) remains failed:
its differing snapshots were not saved, so this diagnostic does not retroactively
prove what every field contained in that run. A new qualification must save both
snapshots and allow only these two source-verified name transitions, preserving
every payload, structural, ownership, allocation and resource check.

The first AMD launch script had a newline-escaping syntax error before any remote
artifact or process was created; its payload/script remain in `amd`. The corrected
`amd-v2` script is compiled before execution. Both successful diagnostic builds
retain two platform-analysis warnings for affinity access. Local contract replay
was separately refused at memory preflight, before any child/inference launched;
that unused attempt is preserved and supplies no qualification.

[Complete observations](observations-20260920.json) retain every metadata difference,
resource total and output identity. All source, binaries, models, runtime pins,
snapshots and samples are under `artifacts/whisper-weight-metadata-20260920`.
No product source or numerical tolerance changed.
'''
    with report.open('x',encoding='utf-8') as f:f.write(text)
    snapshot=BASE/'closure-snapshots';snapshot.mkdir()
    for p in sorted(tracked.iterdir()):
        if p.is_file():shutil.copyfile(p,snapshot/p.name)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    files.update({p.relative_to(ROOT).as_posix():pin(p) for p in [report,data]})
    write(BASE/'closed.json',dict(passed=True,files=files,births=births))
    for name,wanted in files.items():assert pin(ROOT/name)==wanted,name
    print(json.dumps(dict(passed=True,closure=pin(BASE/'closed.json'),pins=len(files),calls=16,original_initializer_checks=original_checks,exact_output_pairs=output_pairs)))


if __name__=='__main__':main()
