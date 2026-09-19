from pathlib import Path
import hashlib,json,os,signal,subprocess,time,traceback,sys
import campaign_processes as accounting
base=Path(__file__).resolve().parent;root=base.parents[1];phase=sys.argv[1];assert phase in ('qual','model')
os.sched_setaffinity(0,{0});out=base/('result-'+phase);assert not out.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    for name,pin in json.loads((base/'bundle.json').read_text())['files'].items():
        p=base/name;assert p.stat().st_size==pin['bytes'] and sha(p)==pin['sha256'],name
def members(group):
    result=[]
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit():continue
        try:
            line=(directory/'stat').read_text();fields=line[line.rfind(')')+2:].split()
            if int(fields[2])!=group or fields[0]=='Z':continue
            status=dict(v.split(':',1) for v in (directory/'status').read_text().splitlines() if ':' in v)
            result.append(dict(pid=int(directory.name),start=int(fields[19]),name=line[line.find('(')+1:line.rfind(')')],
                rss=int(status.get('VmRSS','0 kB').split()[0])*1024,affinity=status['Cpus_allowed_list'].strip()))
        except (FileNotFoundError,ProcessLookupError):pass
    return result
def stop(child,birth):
    if child.poll() is None:
        root=next((m for m in members(child.pid) if m['pid']==child.pid),None)
        assert root is not None and root['start']==birth,'Cannot verify owned process before stopping'
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    return child.wait()

meta=json.loads((base/'provenance.json').read_text())
if phase=='model':
    assert len(sys.argv)==3 and sha(base/'qualification-audit.json')==sys.argv[2]
    gate=json.loads((base/'qualification-audit.json').read_text())
    assert gate['passed'] and gate['phase']=='qual' and gate['core_sha256']==meta['core_sha256'] and gate['product_codegen_inspected']
    prior=json.loads((base/'result-qual/identity.json').read_text())
    assert gate['identity_sha256']==sha(base/'result-qual/identity.json') and gate['bundle_manifest_sha256']==sha(base/'bundle.json')
    assert prior['complete'] and (base/'complete-qual.txt').read_text().strip()=='0' and not Path('/proc',str(prior['supervisor'])).exists()
else:assert len(sys.argv)==2
out.mkdir();line=Path('/proc/self/stat').read_text();birth=int(line[line.rfind(')')+2:].split()[19])
identity=dict(phase=phase,supervisor=os.getpid(),supervisor_start=birth,supervisor_affinity=sorted(os.sched_getaffinity(0)),runs=[],complete=False,
    gate_sha256=sha(base/'qualification-audit.json') if phase=='model' else None)
def save():(out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
clean={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
code=2
try:
    verify();save()
    for pin in meta['assets'].values():
        p=root/pin['remote'];assert p.stat().st_size==pin['bytes'] and sha(p)==pin['sha256'],p
    (out/'cpuinfo.txt').write_text(Path('/proc/cpuinfo').read_text());(out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet','--info'],text=True))
    jobs=meta['qualification'] if phase=='qual' else meta['schedule']
    for index,job in enumerate(jobs):
        tag=job['tag'] if phase=='qual' else f"{index:02d}-{job['name']}-{job['role']}"
        kind=job['kind'] if phase=='qual' else 'model';directory=out/tag;directory.mkdir()
        flags=job['flags'].copy() if phase=='qual' else ({'LOKAD_ONNX_SOFTMAX_ZERO_BLOCKS':'1'} if job['role']=='candidate' else {})
        if kind=='codegen':flags.update(DOTNET_JitDisasm='*SoftmaxMaskedFloatSpanPtrZeroBlocks*',DOTNET_JitStdOutFile=str(directory/'codegen.txt'))
        env=clean|flags;limit=(8 if kind in ('tests','shared') else 6)*1024**3
        if kind=='tests':
            filt='FullyQualifiedName~Softmax|FullyQualifiedName~GraphFusionMaskedSoftmax|FullyQualifiedName~GraphOwnershipTests|FullyQualifiedName~GraphLifetimeTests|FullyQualifiedName~GraphIsolationTests|FullyQualifiedName~GraphBufferReuseTests|FullyQualifiedName~ExecutionIsolationTests|FullyQualifiedName~GraphPreparationConcurrencyTests'
            command=['taskset','-c','2','dotnet','vstest',str(base/'frozen/Lokad.Onnx.Backend.Tests.dll'),'--TestCaseFilter:'+filt,'--logger:trx;LogFileName=tests.trx','--ResultsDirectory:'+str(directory)]
        elif kind=='shared':command=['taskset','-c','2','dotnet',str(base/'shared/bin/Replay.dll'),str(root),str(root/'artifacts/shared-regression-20260918/reference'),str(directory/'managed.json')]
        else:
            case='e5-30pad128' if kind=='codegen' else job['name'];role='codegen' if kind=='codegen' else job['role']
            command=['taskset','-c','2','dotnet',str(base/'model/bin/ProductComparison.dll'),str(root),case,str(root/meta['oracles'][case]['remote']),str(directory/'model.json'),'memory',role]
        row=dict(index=index,tag=tag,kind=kind,job=job,flags=flags,started=time.time(),samples=[],members={},code=None,limit_rss=limit)
        pre=accounting.snapshot();(directory/'pre.json').write_text(json.dumps(pre));(directory/'cpu-pre.txt').write_text(Path('/proc/stat').read_text())
        start=time.monotonic();birth=None
        with (directory/'log.txt').open('x') as log:
            child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            row.update(pid=child.pid,command=command);identity['runs'].append(row);save()
            try:
                while True:
                    found=next((m for m in members(child.pid) if m['pid']==child.pid),None)
                    if found is not None:birth=found['start']
                    if found is not None and found['name']=='dotnet' and found['affinity']=='2':break
                    assert child.poll() is None and time.monotonic()-start<10,'Worker startup failed';time.sleep(.02)
                row['start_identity']=birth
                while child.poll() is None:
                    group=members(child.pid)
                    for m in group:
                        assert m['affinity']=='2',m;row['members'][str(m['pid'])]=dict(start=m['start'],affinity=m['affinity'],name=m['name'])
                    row['samples'].append(dict(seconds=time.monotonic()-start,members=group));save()
                    assert sum(m['rss'] for m in group)<limit and time.monotonic()-start<600,'Resource guard';time.sleep(.25)
            finally:row['code']=stop(child,birth);row.update(seconds=time.monotonic()-start,ended=time.time());save()
        post=accounting.snapshot();(directory/'post.json').write_text(json.dumps(post));(directory/'cpu-post.txt').write_text(Path('/proc/stat').read_text())
        row['accounting']=accounting.foreign_fraction(pre,post,os.getpid());save()
        assert row['code']==0 and not members(child.pid) and not Path('/proc',str(child.pid)).exists(),row
        verify();print(tag,'complete',row['seconds'],flush=True)
    identity['complete']=True;save();code=0
except BaseException:identity['error']=traceback.format_exc();save();traceback.print_exc()
finally:(base/('complete-'+phase+'.txt')).write_text(str(code)+'\n')
raise SystemExit(code)
