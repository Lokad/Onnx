"""Close the complete archived product qualification, preserving the first failed payload."""
from pathlib import Path
from unittest.mock import patch
import argparse,copy,hashlib,json,shutil,subprocess,tarfile,time
import audit as checks

def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args()
    base=a.artifact.resolve();root=Path(__file__).resolve().parents[3];source=Path(__file__).resolve().parent
    payload=base/'collected';meta=checks.read(payload/'frozen.json');state=checks.read(payload/'result-amd/identity.json')
    assert not (base/'closed.json').exists() and not (source/'results-20260920.md').exists()
    result=checks.audit(payload,root,'amd');assert result==checks.read(base/'amd-audit.json')
    check=checks.read(base/'collection-check.json');collection=checks.read(payload/'collection.json')
    assert check['passed'] is True and check['remote']['collection']==collection
    assert checks.pin(base/'results.tar.gz')==check['remote']['archive'] and checks.pin(payload/'collection.json')==check['remote']['receipt']
    assert {p.relative_to(payload).as_posix() for p in payload.rglob('*') if p.is_file()}==set(collection['files'])|{'collection.json'}
    for name,wanted in collection['files'].items():assert checks.pin(payload/name)==wanted,name
    for name,wanted in meta['files'].items():assert checks.pin(base/'payload'/name)==wanted,name
    assert checks.pin(base/'payload/frozen.json')==checks.pin(payload/'frozen.json')
    original=checks.read(base/'source-identity.json')
    assert original['revision']==meta['source_revision'] and original['archive']==meta['source_archive']==checks.pin(base/'source.tar')
    tree=subprocess.check_output(['git','ls-tree','-r',meta['source_revision']],cwd=root,text=True)
    objects={line.split('\t',1)[1]:line.split('\t',1)[0].split()[2] for line in tree.splitlines()}
    with tarfile.open(base/'source.tar') as tar:
        assert tar.pax_headers['comment']==meta['source_revision']
        archived={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    assert set(archived)==set(original['files'])
    normalized=[]
    for name,wanted in original['files'].items():
        data=archived[name]
        raw=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        if raw!=objects[name]:
            # Git archive honors Windows text conversion. Only CRLF-to-LF may differ.
            assert b'\0' not in data and b'\r\n' in data,name
            canonical=data.replace(b'\r\n',b'\n')
            assert hashlib.sha1(b'blob '+str(len(canonical)).encode()+b'\0'+canonical).hexdigest()==objects[name],name
            normalized.append(name)
        assert checks.pin(base/'source'/name)==wanted and checks.pin(payload/name)==wanted
    assert not subprocess.check_output(['git','diff','a34720e',meta['source_revision'],'--','src'],cwd=root,text=True).strip()
    for kind in ['cli','backend','tensors','replay']:
        log=(base/('build-'+kind+'.log')).read_text()
        assert '0 Warning(s)' in log and '0 Error(s)' in log
        assert checks.pin(payload/meta['paths'][kind]/'Lokad.Onnx.dll')==meta['core']
    # Corrupt real in-memory records; immutable evidence files are never modified.
    target=payload/'result-amd/e5-1/output/result.json';real_read=checks.read;good=real_read(target)
    changes=[('completion',lambda v:v.update(passed=False)),('switch',lambda v:v.update(enabled=False)),
        ('core',lambda v:v.update(core_sha256='0'*64)),('flags',lambda v:v['flags'].update(DOTNET_JitOSR='0')),
        ('missing-row',lambda v:v['rows'].pop()),('output-name',lambda v:v['rows'][0].update(name='wrong')),
        ('output-sha',lambda v:v['rows'][0].update(sha256='0'*64)),('reference-sha',lambda v:v['rows'][0].update(reference_sha256='0'*64)),
        ('shape',lambda v:v['rows'][0]['shape'].__setitem__(1,7)),('failed-values',lambda v:v['rows'][0].update(failed_values=1)),
        ('error',lambda v:v['rows'][0].update(max_scaled_error=1.)),('cache',lambda v:v['graphs'][0].update(entries=0)),
        ('inputs',lambda v:v.update(inputs_unchanged=False)),('ownership',lambda v:v.update(held_outputs_unchanged=False))]
    refused=[]
    for label,change in changes:
        damaged=copy.deepcopy(good);change(damaged)
        def reader(path):return damaged if path==target else real_read(path)
        with patch.object(checks,'read',side_effect=reader):
            try:checks.arrays(payload/'result-amd',root,'e5',1,meta)
            except AssertionError:refused.append(label)
            else:raise AssertionError('Damaged record accepted: '+label)
    # State-level corruptions exercise schedule, completion, resources and child identity checks.
    identity=payload/'result-amd/identity.json'
    mutations=[('schedule',lambda v:v['runs'].pop()),('incomplete',lambda v:v.update(complete=False)),
        ('resource-limit',lambda v:v['limits'].update(rss=1)),('sample-count',lambda v:v['runs'][0].update(samples=0)),
        ('peak-rss',lambda v:v['runs'][0].update(peak_rss=0)),('birth',lambda v:v['runs'][0]['child'].update(birth=0))]
    for label,change in mutations:
        damaged=copy.deepcopy(state);change(damaged)
        def reader(path):return damaged if path==identity else real_read(path)
        with patch.object(checks,'read',side_effect=reader):
            try:checks.audit(payload,root,'amd')
            except AssertionError:refused.append(label)
            else:raise AssertionError('Damaged state accepted: '+label)
    # The first attempt is a failed payload, not a failed cache implementation.
    failed=root/'artifacts/e5-fingerprint-product-20260920';prior=checks.read(failed/'closed.json')
    assert prior['verdict']=='FAILED: CLI binaries omitted'
    assert {p.relative_to(failed).as_posix() for p in failed.rglob('*') if p.is_file()}==set(prior['files'])|{'closed.json'}
    for name,wanted in prior['files'].items():assert checks.pin(failed/name)==wanted,name
    births=collection['births']+checks.read(failed/'collected/collection.json')['births']
    assert {tuple(sorted(b.items())) for b in collection['births']}=={tuple(sorted(b.items())) for b in result['births']}
    script="""import sys,json,time
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
rows=json.loads(%r)
for row in rows:
 try:assert psutil.Process(row['pid']).create_time()!=row['birth'],row
 except psutil.NoSuchProcess:pass
print(json.dumps(dict(checked_at=time.time(),births=rows)))
""" % json.dumps(births)
    response=subprocess.run(['ssh','-i','C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem','-o','BatchMode=yes','vermorel@74.178.91.76','python3 -B -'],input=script,text=True,capture_output=True,check=True)
    shutil.copytree(source,base/'closure-source',ignore=shutil.ignore_patterns('bin','obj','__pycache__','results-20260920.md','observations-20260920.json'))
    verification=dict(passed=True,audit=checks.pin(base/'amd-audit.json'),source_files_verified=len(archived),
        source_revision=meta['source_revision'],git_crlf_normalized_files=normalized,damaged_records_refused=refused,terminal=json.loads(response.stdout),
        preserved_failure=checks.pin(failed/'closed.json'))
    write(base/'verification.json',verification)
    files={p.relative_to(base).as_posix():checks.pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
    write(base/'closed.json',dict(schema=1,closed_at=time.time(),verdict='PASS: product correctness; timing pending',files=files))
    receipt=checks.pin(base/'closed.json')
    observations=dict(audit=result,verification=verification,closed=receipt,
        graphs={mode:{str(enabled):checks.read(payload/f'result-amd/{mode}-{enabled}/output/result.json')['graphs'] for enabled in [0,1]} for mode in ['e5','shared']})
    write(source/'observations-20260920.json',observations)
    tests=result['tests'];replay=result['replay']
    table='\n'.join(f"| {kind} | {tests[kind+'-0']['passed']} | {tests[kind+'-1']['passed']} |" for kind in ['backend','tensors'])
    resources=result['resources'];peak=max(r['peak_rss'] for r in resources);minimum=min(r['minimum_available'] for r in resources)
    report=f'''# Exact graph fingerprint cache: product correctness — September20,2026

The actual default-off cache passes full AMD contracts and complete e5/shared-model
output checks. Every on/off output array is byte-identical. This qualifies the
implementation's tested behavior; it supplies no whole-model timing or new ORT
ratio. `LOKAD_ONNX_FINGERPRINT_STRINGS` remains disabled by default.

Source `{meta['source_revision']}` was archived before building. Every one of its
{len(archived)} archived files was independently matched to its Git object
(accounting only for Git's CRLF conversion on{len(normalized)}text files), and
production sources are unchanged from implementation `a34720e`. CLI, backend,
tensor and replay projects build from that extracted source with SDK10.0.204,
zero warnings/errors, and the same core DLL. AMD EPYC9V74 runs .NET10.0.8 on CPU2.

| Full suite | Disabled passed | Enabled passed |
|---|---:|---:|
{table}

Each backend run has3129tests total, including all nine new fingerprint-cache
cases; skipped hardware-fallback cases remain in the TRX records. Earlier Windows
suites passed3036backend tests (93skips) and342tensor tests per setting. Their
initial optional-parameter helper failures were fixed before the final runs and
remain preserved with those original local results.

E5 checks all five lengths8/30/padded128/128/512, Default/Memory and facade/explicit
contexts. Each combination runs twice and again after a harmless node-name edit:
{replay['e5']['on']['arrays']} complete arrays and{replay['e5']['on']['values']:,}values per setting.
The maximum native scaled error is{replay['e5']['on']['max_scaled_error']:.12g}, below the unchanged1e-4gate.
All cached/original fingerprint bits match, intended cache states are observed,
and missing-input failures preserve held outputs. Inputs and every retained output
remain unchanged through later executions and resets.

DINOv3, ResNet50 and GPT-2 replay every existing native reference, including carried
decoder state: {replay['shared']['on']['arrays']}arrays/{replay['shared']['on']['values']:,}values per setting,
maximum scaled error{replay['shared']['on']['max_scaled_error']:.12g}. All off/on bytes match.
The references are retained ORT outputs; no native timing or inference is added.
These shared models do not qualify the unrelated audio intermediate-tensor failures.

All eight workers and their supervisor are terminal. Each inherited CPU2 before
CLR startup; the supervisor used CPU0. Limits were600seconds,12GiB group RSS and
1GiB minimum available memory. All{sum(r['samples'] for r in resources)}resource samples remain;
observed peak group RSS{peak:,}bytes and minimum available{minimum:,}bytes.
No forced GC or runtime override other than the candidate switch was used by
the replay. Tests are functional checks, not performance observations.

The first payload omitted the CLI build. Its disabled backend worker passed3069,
skipped9 and failed51CLI-dependent tests with the explicit missing-build message;
the supervisor stopped before later workers. That entire attempt is independently
closed at `{verification['preserved_failure']['sha256']}`. The corrected payload
includes CLI binaries from the same archive and retains the original criteria.

Independent auditing re-read every exported float, all frozen sources/binaries,
TRX outcomes, exact collection inventory and resource samples. {len(refused)}damaged real
records are refused. [Complete observations](observations-20260920.json) retain
per-worker summaries, exact fingerprints, identities and the closure verification.
The initial read-only closure check expected raw Git blob bytes and refused the
Windows archive's CRLF text. That failed check is preserved; canonical source
comparison resolves it without changing archived bytes or repeating inference.

| Evidence | SHA256 |
|---|---|
| Product core,726016bytes | `{meta['core']['sha256']}` |
| Source archive | `{meta['source_archive']['sha256']}` |
| Frozen payload manifest | `{checks.pin(payload/'frozen.json')['sha256']}` |
| Closed correctness receipt | `{receipt['sha256']}` |

Artifact `{base.name}` binds{len(files)}files plus its closed receipt. Completed
writers must not be rerun into it. [Protocol and tools](README.md) describe
reproduction into a new directory. A separately declared complete-model comparison
and the program's timing qualification remain necessary before default promotion.
'''
    (source/'results-20260920.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(passed=True,closed=receipt,files=len(files),damaged_records_refused=len(refused))))

if __name__=='__main__':main()
