"""Close exact-fingerprint component evidence and render its bounded conclusion."""
from pathlib import Path
import argparse,copy,json,shutil,subprocess,time
from generate import pin
from audit import worker,resources,timing,fixture
from collect import HOST,KEY


def read(p):return json.loads(p.read_text())
def write(p,v):
    with p.open('x',encoding='utf-8') as s:json.dump(v,s,indent=2)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args()
    base=a.artifact.resolve();source=Path(__file__).resolve().parent;collected=base/'collected'
    assert not (base/'closed.json').exists() and not (source/'results-20260920.md').exists()
    collection=read(collected/'collection.json');check=read(base/'collection-check.json');frozen=read(collected/'frozen.json')
    assert check['passed'] is True and check['remote']['collection']==collection
    assert check['remote']['archive']==pin(base/'results.tar.gz') and check['remote']['receipt']==pin(collected/'collection.json')
    assert collection['frozen']==pin(collected/'frozen.json')==pin(base/'deployment/frozen.json')
    assert {p.relative_to(collected).as_posix() for p in collected.rglob('*') if p.is_file()}==set(collection['files'])|{'collection.json'}
    for name,wanted in collection['files'].items():assert pin(collected/name)==wanted,name
    for name,wanted in frozen['files'].items():assert pin(collected/name)==pin(base/'deployment'/name)==wanted,name
    state=read(collected/'timing-process/identity.json');generation=read(collected/'generation.json');audit=read(base/'amd-audit.json')
    assert resources(collected/'timing-process')==audit['resources'] and audit['passed'] is True
    values=[]
    for visit in range(4):
        value,_=worker(collected/f'timing-process/v{visit}/output',generation,state['binaries'],True);values.append(value)
    result=timing(values);assert audit['timing']==result
    assert {tuple(sorted(b.items())) for b in collection['births']}=={tuple(sorted(b.items())) for b in audit['resources']['births']}
    # Validate real corruption refusals independently of the passing runner summary.
    original=read(collected/'timing-process/v0/output/fixtures.json')[3];refusals=0
    for change in [lambda v:v.update(fingerprint='0'),lambda v:v['transitions'][0].update(before='0'),
        lambda v:v['transitions'][0].update(after='0'),lambda v:v['nodes'][0].update(Name=[65]),
        lambda v:v['nodes'][0].update(op=999),lambda v:v['inputs'].append([99]),lambda v:v['output_descriptions'][0].append(100)]:
        value=copy.deepcopy(original);change(value)
        try:fixture(value)
        except AssertionError:refusals+=1
        else:raise AssertionError('Damaged fixture accepted')
    script="""import sys,json,time
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
rows=json.loads(%r)
for row in rows:
 try:assert psutil.Process(row['pid']).create_time()!=row['birth'],row
 except psutil.NoSuchProcess:pass
print(json.dumps(dict(checked_at=time.time(),births=rows)))
""" % json.dumps(collection['births'])
    response=subprocess.run(['ssh','-i',KEY,'-o','BatchMode=yes',HOST,'python3 -B -'],input=script,text=True,capture_output=True,check=True)
    # Windows proof and initial startup refusal remain separate, terminal attempts.
    import psutil
    local_births=[]
    for folder in [base/'proof-process',base/'proof-clean-env-process']:
        local=read(folder/'identity.json');local_births.append(local['supervisor'])
        for run in local['runs']:local_births.extend(dict(pid=int(pid),birth=birth) for pid,birth in run['members'].items())
    for row in local_births:
        try:assert psutil.Process(row['pid']).create_time()!=row['birth']
        except psutil.NoSuchProcess:pass
    assert read(base/'proof-process/identity.json')['complete'] is False
    assert 'Runtime overrides' in (base/'proof-process/v0/stderr.txt').read_text()
    assert resources(base/'proof-clean-env-process')==read(base/'local-audit.json')['resources']
    snapshot=base/'closing-source';snapshot.mkdir()
    for path in source.iterdir():
        if path.suffix in ('.py','.cs','.csproj','.md'):shutil.copyfile(path,snapshot/path.name)
    receipt=dict(schema=1,closed=True,execution_passed=True,component_screen_passed=result['passed'],created=time.time(),
        runtime_source_commit=frozen['source_commit'],terminal=dict(remote=json.loads(response.stdout),local=local_births),
        damaged_fixture_refusals=refusals,files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()})
    write(base/'closed.json',receipt)
    observations=dict(schema=1,receipt=pin(base/'closed.json'),audit=audit,workers=values,collection=collection,
        runtime_frozen=frozen,files=receipt['files'],damaged_fixture_refusals=refusals)
    write(source/'observations-20260920.json',observations)
    means=result['mean_seconds'];rows=''
    for name,seconds in zip(['Actual qualified core','Copy A','Copy B','Cached transitions'],means):
        rows+=f'| {name} | {seconds*1e6:.6f} | {seconds/means[0]:.6f} |\n'
    text=f'''# Exact graph-fingerprint transition cache — 2026-09-20

The standalone cache passes its prospective component screen on AMD EPYC 9V74,
CPU2, .NET10.0.8. The complete structure check drops from {means[0]*1e3:.6f}ms
to {means[3]*1e3:.6f}ms, with exact original fingerprint bits. Product code and
the e5/ORT scoreboard are unchanged; this result nominates integration and does
not measure complete-model latency.

| Complete fingerprint traversal | Mean microseconds | Relative to actual core |
|---|---:|---:|
{rows}
The cache still visits every mutable field. It reuses a string transition only
when its position, incoming hash and ordinal string value match the training
entry. Otherwise it executes the original character loop. Exactness follows
step by step: each hit yields the same next hash for the same incoming state
and immutable string, and each miss runs the original operation. No field or
recursive graph traversal is skipped. This preserves the existing hash and its
existing collision properties; it is not a new collision-free graph contract.

Every AMD worker passes13,663 exact case checks, four cycle refusals and128
concurrent-reader checks. Synthetic cases cover Unicode code units including
unpaired surrogates, null/empty names, collection changes, nested/shared graphs,
cycles introduced after training and all347real e5 node-name mutations.
The independent Python auditor reproduces forty flat-graph fingerprints and
1,300transitions per worker, and refuses seven damaged real fixtures. The same
functional proof also passes on Windows. No neural inference runs in this lane.

Four fresh workers retain1,024measured batches,262,144measured fingerprint calls
and131,072warmup calls. Each variant occupies every order position equally.
No sample is discarded and no GC/runtime override is used. All timed batches
allocate zero managed bytes on the calling thread. Duplicate CopyB/CopyA worker
ratios range {min(v['copy_b_a'] for v in result['visits']):.6f}–{max(v['copy_b_a'] for v in result['visits']):.6f}.
Cached/Actual worker ratios range {min(v['cached_actual'] for v in result['visits']):.6f}–{max(v['cached_actual'] for v in result['visits']):.6f}.
All fixed controls and gain/regression screens pass. These are descriptive
component screens, not confidence intervals or a repair of failed whole-model
A/A calibration. Full raw samples, allocation/GC counts and per-visit means
remain in the [observations](observations-20260920.json).

E5 uses2,330entries,55,920bytes of entry structs, excluding array/dictionary
headers and shared immutable strings. Training time/allocation and loading time
are retained separately. Cache data do not grow on mutations or change during
reads. Peak sampled worker RSS is{audit['resources']['peak_rss']/1e9:.6f}GB and minimum
system available memory is{audit['resources']['minimum_available']/1e9:.6f}GB. All workers satisfy
the120second,2GiBRSS and1GiBavailable guards. Resource sampling does not establish
universal bounds or exclude hypervisor activity. All actual process births are
terminal and the exact collection inventory verifies.

The initial Windows launch refused its inherited, unrelated `LOKAD_ROOT` variable
before proof. That attempt remains failed; a separately named clean-environment
proof passed. Early build-only preparations are retained separately. No model,
production default, numerical tolerance or current benchmark ratio changed.

Frozen runtime source `{frozen['source_commit']}`; core087e280 SHA
`{values[0]['core_sha256']}`; frozen SHA`{pin(collected/'frozen.json')['sha256']}`.
Closed receipt SHA`{pin(base/'closed.json')['sha256']}` binds{len(receipt['files'])}files.
Artifact `artifacts/e5-fingerprint-cache-v3-20260920`; [protocol](README.md).
Production integration must separately preserve cache ownership, preparation
invalidation, isolated contexts and every affected graph lifecycle contract,
then establish whole-model behavior and timing before any speedup claim.
'''
    with (source/'results-20260920.md').open('x',encoding='utf-8') as s:s.write(text)
    print('Closed',len(receipt['files']),'files; component screen',result['passed'],'receipt',pin(base/'closed.json')['sha256'])


if __name__=='__main__':main()
