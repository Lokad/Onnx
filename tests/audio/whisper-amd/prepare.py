"""Build only the unchanged timing consumer and bind qualified integrated source."""
import importlib.util,json,os,shutil,subprocess,time,traceback
from common import *

def main():
    assert read(PRODUCT/'final-verification.json')['passed']
    assert read(CONTRACTS/'final-verification.json')['passed']
    assert read(INTEGRATION/'applied.json')['applied']
    assert not BASE.exists();BASE.mkdir();source=BASE/'source';source.mkdir()
    inventory=read(PRODUCT/'source-manifest.json');matched={}
    for name,wanted in inventory.items():
        if name.startswith('src/') or name in ['global.json','Lokad.Onnx.slnx']:
            qualified=PRODUCT/'source'/name;assert pin(qualified)==wanted
            assert (ROOT/name).read_bytes().replace(b'\r\n',b'\n')==qualified.read_bytes().replace(b'\r\n',b'\n'),name
            matched[name]=pin(ROOT/name)
    for name in ['tests/audio/whisper-comparison/Program.cs','tests/audio/whisper-comparison/WhisperBenchmark.csproj','tests/Shared/NpySupport.cs','global.json']:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    # Bind qualification receipts and the exact source bridge, without replaying tests.
    proofs={str(p.relative_to(ROOT)):pin(p) for p in [PRODUCT/'closed.json',PRODUCT/'final-verification.json',CONTRACTS/'closed.json',CONTRACTS/'final-verification.json',INTEGRATION/'applied.json',INTEGRATION/'source-equivalence.json']}
    sys.path.insert(0,str(LOCAL_SITE));import psutil
    own=psutil.Process();own.cpu_affinity([0]);members={};child=None;started=time.monotonic()
    command=['dotnet','build','WhisperBenchmark.csproj','-c','Release','--tl:off','--nologo','-v','minimal','--disable-build-servers','-m:1','-p:FrozenProductDirectory='+str(BIN)]
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    env.update(DOTNET_PROCESSOR_COUNT='1',UseSharedCompilation='false',MSBUILDDISABLENODEREUSE='1',DOTNET_CLI_DO_NOT_USE_MSBUILD_SERVER='1')
    state=dict(command=command,complete=False,code=None,members=members)
    try:
        with (BASE/'build.stdout').open('x') as out,(BASE/'build.stderr').open('x') as err,(BASE/'build.samples.jsonl').open('x') as samples:
            child=subprocess.Popen(command,cwd=source/'tests/audio/whisper-comparison',env=env,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
            process=psutil.Process(child.pid);state['child']=dict(pid=child.pid,birth=process.create_time());members[str(child.pid)]=state['child']['birth']
            while child.poll() is None:
                live=[]
                try:
                    assert process.create_time()==state['child']['birth']
                    for p in [process]+process.children(recursive=True):
                        try:
                            b=p.create_time();assert members.get(str(p.pid),b)==b;members[str(p.pid)]=b
                            live.append(dict(pid=p.pid,birth=b,rss=p.memory_info().rss,affinity=p.cpu_affinity()))
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:pass
                row=dict(seconds=time.monotonic()-started,available=psutil.virtual_memory().available,members=live)
                samples.write(json.dumps(row)+'\n');samples.flush()
                assert row['seconds']<300 and row['available']>=1024**3 and sum(p['rss'] for p in live)<4*1024**3
                assert all(p['affinity']==[0] for p in live);time.sleep(.25)
            state['code']=child.wait();assert state['code']==0
    except BaseException:
        state['error']=traceback.format_exc()
        for pid,birth in reversed(list(members.items())):
            try:
                p=psutil.Process(int(pid))
                if p.create_time()==birth:p.kill()
            except psutil.NoSuchProcess:pass
        if child is not None:child.wait(timeout=10)
        raise
    finally:
        state.update(complete=True,seconds=time.monotonic()-started);write(BASE/'build.json',state)
    for pid,birth in members.items():
        try:assert psutil.Process(int(pid)).create_time()!=birth
        except psutil.NoSuchProcess:pass
    built=source/'tests/audio/whisper-comparison/bin/Release/net10.0';(BASE/'bin').mkdir()
    names=['WhisperBenchmark.dll','WhisperBenchmark.deps.json','WhisperBenchmark.runtimeconfig.json']
    names+=['Lokad.Onnx.dll','Lokad.Onnx.Data.dll','Google.Protobuf.dll','FastBertTokenizer.dll','Lokad.Tokenizers.dll','SixLabors.ImageSharp.dll']
    for name in names:
        if not name.startswith('WhisperBenchmark.'):assert pin(built/name)==pin(BIN/name)
        shutil.copyfile(built/name,BASE/'bin'/name)
    prior=PRIOR/'collected';frozen=read(prior/'frozen.json');runs=read(prior/'campaign/identity.json')['runs']
    run=next(r for r in runs if r['family']=='whisper' and r['engine']=='ort');assert run['phase']=='conformance' and run['code']==0
    spec=importlib.util.spec_from_file_location('original_audio_audit',ROOT/'tests/audio/amd-comparison/audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    manifest=read(prior/'manifests/whisper.json');folder=prior/run['output'];value=read(folder/'worker/result.json')
    validate_records(value,manifest,'conformance');audit.worker_identity(value,manifest,frozen,prior,'ort')
    resources=audit.resource_records(run,[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()])
    for i,row in enumerate(value['records']):assert read(folder/'worker'/f'{i:03}.json')==row
    gate=dict(passed=True,calls=20,worker=run,resource=resources,result=pin(folder/'worker/result.json'),manifest=pin(prior/'manifests/whisper.json'),frozen=pin(prior/'frozen.json'),closure=pin(PRIOR/'failure-closed.json'))
    assert read(PRIOR/'failure-closed.json')['closure_passed'];write(BASE/'prior-native-gate.json',gate)
    updated=dict(manifest);revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    updated.update(product_source=revision,core_sha256=pin(BASE/'bin/Lokad.Onnx.dll')['sha256'],data_sha256=pin(BASE/'bin/Lokad.Onnx.Data.dll')['sha256'])
    (BASE/'manifests').mkdir();write(BASE/'manifests/whisper.json',updated)
    shutil.copyfile(ROOT/'.agent/m5-whisper-amd-timing-20260921.md',BASE/'prospective-plan.md')
    write(BASE/'prepared.json',dict(passed=True,source=revision,product_source=revision,qualified=proofs,source_bridge=matched,build=pin(BASE/'build.json'),prior_native=gate,bin={n:pin(BASE/'bin'/n) for n in names},manifest=pin(BASE/'manifests/whisper.json')))
    print(json.dumps(dict(prepared=True,source=revision,product_files=len(matched),build_seconds=state['seconds'],receipt=pin(BASE/'prepared.json'))))

if __name__=='__main__':main()
