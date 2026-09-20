"""Independently close both host qualifications after every worker has exited."""
from pathlib import Path
from unittest.mock import patch
import argparse,copy,hashlib,json,subprocess,tarfile,time
import audit as checks

def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args()
    base=a.artifact.resolve();root=Path(__file__).resolve().parents[3];source=Path(__file__).resolve().parent
    assert not (base/'closed.json').exists() and not (source/'results-20260920.md').exists()
    meta=checks.read(base/'payload/frozen.json');results={}
    for label,folder in [('windows','payload'),('amd','collected')]:
        results[label]=checks.audit(base/folder,root,label)
        assert results[label]==checks.read(base/(label+'-audit.json'))
    collection=checks.read(base/'collected/collection.json');transfer=checks.read(base/'collection-check.json')
    assert transfer['passed'] and transfer['remote']['collection']==collection
    assert checks.pin(base/'results.tar.gz')==transfer['remote']['archive']
    assert checks.pin(base/'collected/collection.json')==transfer['remote']['receipt']
    assert {p.relative_to(base/'collected').as_posix() for p in (base/'collected').rglob('*') if p.is_file()}==set(collection['files'])|{'collection.json'}
    for name,wanted in collection['files'].items():assert checks.pin(base/'collected'/name)==wanted,name
    assert checks.pin(base/'payload/frozen.json')==checks.pin(base/'collected/frozen.json')
    original=checks.read(base/'source-identity.json')
    assert original['revision']==meta['source_revision'] and original['archive']==meta['source_archive']==checks.pin(base/'source.tar')
    tree=subprocess.check_output(['git','ls-tree','-r',meta['source_revision']],cwd=root,text=True)
    objects={line.split('\t',1)[1]:line.split('\t',1)[0].split()[2] for line in tree.splitlines()}
    with tarfile.open(base/'source.tar') as tar:
        assert tar.pax_headers['comment']==meta['source_revision']
        archived={m.name:tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    assert set(archived)==set(original['files']);normalized=[]
    for name,wanted in original['files'].items():
        data=archived[name]
        if hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()!=objects[name]:
            assert b'\0' not in data and b'\r\n' in data,name
            canonical=data.replace(b'\r\n',b'\n')
            assert hashlib.sha1(b'blob '+str(len(canonical)).encode()+b'\0'+canonical).hexdigest()==objects[name],name
            normalized.append(name)
        assert checks.pin(base/'source'/name)==wanted and checks.pin(base/'payload'/name)==wanted
    for kind in ['cli','backend','tensors','replay']:
        log=(base/('build-'+kind+'.log')).read_text()
        assert '0 Warning(s)' in log and '0 Error(s)' in log
        assert checks.pin(base/'payload'/meta['paths'][kind]/'Lokad.Onnx.dll')==meta['core']

    # Demonstrate refusals against real records without altering raw evidence.
    target=base/'collected/result-amd/e5-1/output/result.json';real_read=checks.read;good=real_read(target)
    changes=[('completion',lambda v:v.update(passed=False)),('switch',lambda v:v.update(enabled=False)),
        ('actual-switch',lambda v:v.update(actual_switch=False)),('core',lambda v:v.update(core_sha256='0'*64)),
        ('flags',lambda v:v['flags'].update(DOTNET_JitOSR='0')),('missing-row',lambda v:v['rows'].pop()),
        ('output-name',lambda v:v['rows'][0].update(name='wrong')),('output-sha',lambda v:v['rows'][0].update(sha256='0'*64)),
        ('reference-sha',lambda v:v['rows'][0].update(reference_sha256='0'*64)),
        ('shape',lambda v:v['rows'][0]['shape'].__setitem__(1,7)),('failed-values',lambda v:v['rows'][0].update(failed_values=1)),
        ('error',lambda v:v['rows'][0].update(max_scaled_error=1.)),('inputs',lambda v:v.update(inputs_unchanged=False)),
        ('ownership',lambda v:v.update(held_outputs_unchanged=False))]
    refused=[]
    for label,change in changes:
        damaged=copy.deepcopy(good);change(damaged)
        def reader(path):return damaged if path==target else real_read(path)
        with patch.object(checks,'read',side_effect=reader):
            try:checks.arrays(base/'collected/result-amd',root,'e5',1,meta)
            except AssertionError:refused.append(label)
            else:raise AssertionError('Damaged record accepted: '+label)
    code=(base/'collected/result-amd/code-1/jit.txt').read_text()
    for label,bad in [('no-wide-code',code.replace('zmm','ymm')),('wrong-method',code.replace('LayerNormFloatInto','WideOutput')),
                      ('fused-code',code.replace('vmulpd','vfmadd213pd')),('no-optimized-tier',code.replace('Tier1','Tier0').replace('FullOpts','MinOpts'))]:
        try:checks.inspect(bad)
        except AssertionError:refused.append(label)
        else:raise AssertionError('Damaged code accepted: '+label)

    import psutil
    local_births=results['windows']['births']
    for item in local_births:
        try:assert psutil.Process(item['pid']).create_time()!=item['birth'],item
        except psutil.NoSuchProcess:pass
    script="import sys,json\nsys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')\nimport psutil\nitems="+repr(results['amd']['births'])+"\nfor item in items:\n try:assert psutil.Process(item['pid']).create_time()!=item['birth'],item\n except psutil.NoSuchProcess:pass\nprint(json.dumps(dict(all_terminal=True,births=items)))\n"
    response=subprocess.run(['ssh','-i','C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem','-o','BatchMode=yes','vermorel@74.178.91.76','python3 -B -'],input=script,text=True,capture_output=True,check=True)
    verification=dict(passed=True,source_revision=meta['source_revision'],source_files_verified=len(archived),
        git_crlf_normalized_files=normalized,damaged_records_refused=refused,
        windows_terminal=dict(all_terminal=True,births=local_births),amd_terminal=json.loads(response.stdout))
    write(base/'verification.json',verification)
    write(source/'observations-20260920.json',dict(audits=results,verification=verification))
    rows=[]
    for host,result in results.items():
        for kind in ['backend','tensors']:
            off=result['tests'][kind+'-0'];on=result['tests'][kind+'-1']
            rows.append(f"| {host} {kind} | {off['passed']} / {off['total']} | {on['passed']} / {on['total']} |")
    table='\n'.join(rows);amd=results['amd'];win=results['windows']
    code_result=amd['code'];resources=[r for result in results.values() for r in result['resources']]
    report=f'''# Wider LayerNorm: actual product correctness — September 20, 2026

The integrated default-off wider float LayerNorm transform passes complete
Windows and AMD qualification. Every off/on e5 and shared-model output array
is byte-identical on its host. Actual optimized AMD product code contains the
intended double AVX-512 transform, original vector arithmetic and no fused
multiply-add. This supplies no new whole-model timing or native ORT ratio.

Source `{meta['source_revision']}` was archived before building; all
{len(archived)} archived source files match their Git objects (allowing only
Git's recorded CRLF text conversion). CLI, backend tests, tensor tests and
replay build with SDK 10.0.204, zero warnings/errors and the same core DLL:
`{meta['core']['sha256']}`. The Windows i7-14700KF uses .NET 10.0.12 and lacks
hardware Vector512; AMD EPYC 9V74 uses .NET 10.0.8 with hardware Vector512,
AVX-512F and eight-float ordinary vectors. Inference is pinned to logical CPU 2.

| Full suite | Switch off: passed / total | Switch on: passed / total |
|---|---:|---:|
{table}

All 45 new public API cases pass in every backend run: independent references,
vector/tail and real widths, bias/no-bias, guarded buffers, in-place output,
held results, multi-axis geometry and exceptional values. Remaining non-passes
are the retained hardware-specific skips, with no failed tests.

Each host/setting retains 60 complete e5 arrays (five lengths, two lifetime
policies, facade/explicit contexts, three calls) and 106 shared-model arrays
(DINOv3, ResNet50, GPT-2 including carried state). Inputs and retained outputs
survive later calls and missing-input failures. All arrays pass the original
native scaled 1e-4 gate. Maximum e5 errors are
{win['replay']['e5']['on']['max_scaled_error']:.12g} on Windows and
{amd['replay']['e5']['on']['max_scaled_error']:.12g} on AMD; shared maxima are
{win['replay']['shared']['on']['max_scaled_error']:.12g} and
{amd['replay']['shared']['on']['max_scaled_error']:.12g}, respectively.

The separate AMD code job emits {code_result['versions']} kernel versions.
Its selected optimized body is `{code_result['header']}`,
{code_result['code_bytes']} code bytes, SHA256
`{code_result['optimized_sha256']}`. Both statistics passes keep their source
arithmetic and reduction order. The sixteen-float output transform is followed
by the original vector/scalar tails. The code job is instrumented and untimed
for performance purposes; its duration is not a latency result.

All 17 workers finish within their fixed 600-second / 12-GiB RSS / 1-GiB
available-memory limits. The audit retains {sum(r['samples'] for r in resources)}
resource samples, maximum sampled group RSS {max(r['peak_rss'] for r in resources)}
bytes and minimum available memory {min(r['minimum_available'] for r in resources)}
bytes. Every supervisor/observed worker birth is independently terminal. The
auditor rejects {len(refused)} damaged output/switch/ownership/code records.

The first remote preflight mistakenly classified the two Azure guest-agent
services as inference workers and refused launch. Its error is preserved;
after identifying those exact services, the pristine payload was reverified
and launched once. Neither service was terminated and no inference was replayed.

Complete commands and boundaries are in [README.md](README.md); numerical,
test, resource, source and process observations are in
[observations-20260920.json](observations-20260920.json). Raw evidence is
`artifacts/e5-layernorm-product-20260920`; `closed.json` binds its files and
both reports after all writers finish. The switch remains off pending a
separate complete-model comparison. Audio numerical gaps remain open.
'''
    with (source/'results-20260920.md').open('x',encoding='utf-8') as stream:stream.write(report)
    files={p.relative_to(base).as_posix():checks.pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
    reports={p.relative_to(root).as_posix():checks.pin(p) for p in [source/'observations-20260920.json',source/'results-20260920.md',Path(__file__).resolve()]}
    write(base/'closed.json',dict(schema=1,closed_at=time.time(),verdict='PASS: actual product correctness; whole-model timing pending',files=files,reports=reports))
    print(json.dumps(dict(passed=True,receipt=checks.pin(base/'closed.json'),files=len(files),reports=len(reports))))

if __name__=='__main__':main()
