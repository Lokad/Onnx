"""Bounded normal-runtime prototype: conformance, then conditionally endurance."""
from pathlib import Path
import importlib.util, json, os, subprocess, sys, time, traceback
import psutil
from reuse_protocol import validate_reuse,allocation_gate
from sharing_protocol import validate_sharing
from protocol import pin,read,write,LIMITS,schedule,check_sample,validate_records


def absent(pid,birth):
    try:return psutil.Process(pid).create_time()!=birth
    except psutil.NoSuchProcess:return True


def verify(base,frozen):
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    for name,wanted in frozen['external'].items():assert pin(Path(name))==wanted,name


def run(base):
    assert psutil.__version__=='7.0.0' and os.name=='posix'
    own=psutil.Process();own.cpu_affinity([0]);frozen=read(base/'frozen.json')
    assert frozen['limits']==LIMITS;verify(base,frozen)
    (base/'campaign').mkdir();state=dict(complete=False,code=None,started=time.time(),supervisor=dict(pid=own.pid,birth=own.create_time()),
                                       frozen=pin(base/'frozen.json'),limits=LIMITS,runs=[])
    def save():
        target=base/'campaign/identity.tmp';target.write_text(json.dumps(state,indent=2));target.replace(base/'campaign/identity.json')
    accounting_spec=importlib.util.spec_from_file_location('audio_process_accounting',base/'runtime/campaign_processes.py')
    account=importlib.util.module_from_spec(accounting_spec);accounting_spec.loader.exec_module(account)
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    env.update(PYTHONPATH=os.pathsep.join(frozen['python_paths']),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
    env.update({key:'1' for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']})
    campaign_start=time.monotonic();save()
    try:
        assert frozen['scope']=='whisper-weight-sharing-v2' and frozen['conformance_calls']==20 and frozen['endurance_calls']==80 and frozen['explicit_gc'] is False
        for phase in ['conformance','timing']:
            if phase=='timing':assert read(base/'campaign/conformance-gate.json')['passed'] is True
            (base/'campaign'/phase).mkdir()
            for index,(family,engine) in enumerate([('whisper','managed')]):
                assert time.monotonic()-campaign_start<LIMITS['campaign_seconds']
                available=psutil.virtual_memory().available;disk=psutil.disk_usage(str(base)).free
                assert available>=LIMITS['preflight'] and disk>=LIMITS['preflight_disk']
                name=f'{phase}-{index:02}-{family}-{engine}';relative=Path('campaign')/phase/f'{index:02}-{family}-{engine}';out=base/relative;out.mkdir()
                manifest=read(base/'manifests'/(family+'.json'))
                command=(['python3','-B',str(base/'runtime/native.py')] if engine=='ort' else ['dotnet',str(base/'bin'/('WhisperWeightSharingV2.dll' if family=='whisper' else 'AudioBenchmark.dll'))])
                command += [str(base/'assets'),str(base/'manifests'/(family+'.json')),str(out/'worker'),phase]
                run=dict(name=name,phase=phase,family=family,engine=engine,command=command,output=relative.as_posix(),started=time.time(),
                         preflight_available=available,preflight_disk=disk,members={},samples=0,peak_rss=0,complete=False,code=None)
                state['runs'].append(run);save();child=None;start=time.monotonic();before=account.snapshot();write(out/'pre.json',before)
                try:
                    with (out/'stdout.txt').open('x') as stdout,(out/'stderr.txt').open('x') as stderr,(out/'samples.jsonl').open('x') as stream:
                        own.cpu_affinity([2])
                        try:child=subprocess.Popen(command,cwd=base,env=env,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,start_new_session=True)
                        finally:own.cpu_affinity([0])
                        process=psutil.Process(child.pid);run['child']=dict(pid=child.pid,birth=process.create_time())
                        run['members'][str(child.pid)]=run['child']['birth'];save()
                        while child.poll() is None:
                            members=[]
                            try:
                                assert process.create_time()==run['child']['birth']
                                for p in [process]+process.children(recursive=True):
                                    try:
                                        birth=p.create_time();assert run['members'].get(str(p.pid),birth)==birth;run['members'][str(p.pid)]=birth
                                        threads=[]
                                        for thread in p.threads():
                                            try:threads.append(dict(tid=thread.id,affinity=sorted(os.sched_getaffinity(thread.id))))
                                            except ProcessLookupError:pass
                                        members.append(dict(pid=p.pid,birth=birth,rss=p.memory_info().rss,affinity=p.cpu_affinity(),threads=threads))
                                    except psutil.NoSuchProcess:pass
                            except psutil.NoSuchProcess:pass
                            if not members and child.poll() is not None:break
                            row=dict(seconds=time.monotonic()-start,available=psutil.virtual_memory().available,disk=psutil.disk_usage(str(base)).free,members=members)
                            stream.write(json.dumps(row)+'\n');stream.flush();run['samples']+=1;run['peak_rss']=max(run['peak_rss'],sum(m['rss'] for m in members));save()
                            check_sample(row);assert time.monotonic()-campaign_start<LIMITS['campaign_seconds'];time.sleep(.5)
                        run['code']=child.wait();assert run['code']==0
                    deadline=time.monotonic()+10
                    while not all(absent(int(pid),birth) for pid,birth in run['members'].items()):
                        assert time.monotonic()<deadline;time.sleep(.1)
                    value=read(out/'worker/result.json');validate_records(value,manifest,phase);validate_reuse(value);validate_sharing(value)
                    for stage in ['before','after']:assert read(out/'worker'/('weights-'+stage+'.json'))==value['weight_sharing'][stage]
                    assert value['manifest_sha256']==pin(base/'manifests'/(family+'.json'))['sha256']
                    if engine=='managed':
                        assert value['runtime']=='.NET 10.0.8' and value['processor_count']==1
                        for key in ['core_sha256','data_sha256']:assert value[key]==manifest[key]
                        assert value['runner_sha256']==pin(Path(command[1]))['sha256']
                    else:
                        assert value['versions']==manifest['native_versions'] and value['native_binaries']==manifest['native_binaries']
                        assert value['python_binary']==frozen['interpreter']
                        for path,wanted in value['numeric_libraries'].items():assert frozen['external'][path]==wanted==pin(Path(path)),path
                    after=account.snapshot();write(out/'post.json',after);run['accounting']=account.foreign_fraction(before,after,own.pid)
                except BaseException as error:
                    run['error']=repr(error)
                    for pid,birth in reversed(list(run['members'].items())):
                        if not absent(int(pid),birth):
                            try:psutil.Process(int(pid)).kill()
                            except psutil.NoSuchProcess:pass
                    if child is not None:child.wait(timeout=10)
                    raise
                finally:run.update(complete=True,seconds=time.monotonic()-start,ended=time.time());save()
                print(json.dumps(dict(name=name,seconds=run['seconds'],code=run['code'])),flush=True)
                if phase=='conformance':
                    original=[read(base/'original-prefix'/f'{i:03}.json') for i in range(16)]
                    gate=allocation_gate(value,original)
                    gate.update(worker=pin(out/'worker/result.json'),frozen=pin(base/'frozen.json'))
                    write(base/'campaign/conformance-gate.json',gate)
        verify(base,frozen);state['code']=0
    except BaseException as error:state.update(code=1,error=repr(error));traceback.print_exc()
    finally:state.update(complete=True,ended=time.time(),seconds=time.monotonic()-campaign_start);save()
    return state['code']


if __name__=='__main__':sys.exit(run(Path(sys.argv[1]).resolve()))
