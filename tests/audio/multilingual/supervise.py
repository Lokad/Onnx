"""Run the fixed four-worker accuracy schedule with immutable payload and births."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import psutil
from common import CORE,DATA,pin,read,sha,write_new


def verify(root,files):
    for name,wanted in files.items():assert pin(root/name)==wanted,name


def save_status(path,value,timeout=1.):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    deadline=time.monotonic()+timeout
    while True:
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            # Windows readers can temporarily deny FILE_SHARE_DELETE.
            if os.name!='nt' or error.winerror not in (5,32) or time.monotonic()>=deadline:raise
            time.sleep(.01)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[3];base=a.artifact.resolve();out=base/'run';out.mkdir()
    assert psutil.__version__=='7.0.0' and sha(base/'bin/Lokad.Onnx.dll')==CORE and sha(base/'bin/Lokad.Onnx.Data.dll')==DATA
    assert read(base/'input-audit.json')['passed']
    runtime_names=('common.py','native.py','whisper_adapter.py','supervise.py','Program.cs','MultilingualReplay.csproj','dataset.json',
                   'audit.py','scoring.py','audit_inputs.py','prepare.py')
    source_paths=[Path(__file__).parent/name for name in runtime_names]
    source_paths.extend([root/'tests/Shared/NpySupport.cs',root/'eng/campaign_processes.py',root/'tests/audio/comparison/native_adapters.py'])
    files={}
    frozen=base/'runtime-source';frozen.mkdir()
    for source in source_paths:
        target=frozen/source.name;shutil.copyfile(source,target)
        files[source.relative_to(root).as_posix()]=pin(source)
        files[target.relative_to(root).as_posix()]=pin(target)
    for folder in ('bin','inputs','manifests'):
        for path in (base/folder).iterdir():
            if path.is_file():files[path.relative_to(root).as_posix()]=pin(path)
    files[(base/'input-audit.json').relative_to(root).as_posix()]=pin(base/'input-audit.json')
    for family in ('parakeet','whisper'):
        manifest=read(base/'manifests'/(family+'.json'))
        for spec in list(manifest['source_files'].values())+list(manifest['native_binaries'].values()):
            files[spec['path']]={k:spec[k] for k in ('bytes','sha256')}
    shutil.copyfile(root/'.agent/m4-asr-multilingual-20260920.md',base/'prospective-plan.md')
    files[(base/'prospective-plan.md').relative_to(root).as_posix()]=pin(base/'prospective-plan.md')
    snapshot=dict(schema=1,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                  product_source='087e280b5ea0a6a610399ccffd1a1e5668def10e',files=files)
    write_new(base/'frozen.json',snapshot)
    spec=importlib.util.spec_from_file_location('multilingual_accounting',frozen/'campaign_processes.py')
    accounting=importlib.util.module_from_spec(spec);spec.loader.exec_module(accounting)
    parent=psutil.Process();prior=parent.cpu_affinity();parent.cpu_affinity([0])
    identity=dict(schema=1,supervisor=dict(pid=parent.pid,birth=parent.create_time()),started=time.time(),complete=False,runs=[],
                  frozen_sha256=sha(base/'frozen.json'),limits=dict(rss=20*1024**3,seconds=3600,available=1024**3))
    def save():
        save_status(out/'identity.json',identity)
    env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
    env.update(PYTHONUTF8='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    try:
        verify(root,files);save()
        (out/'dotnet-info.txt').write_text(subprocess.check_output(['dotnet','--info'],text=True),encoding='utf-8')
        for number,(family,engine) in enumerate((('parakeet','ort'),('parakeet','managed'),('whisper','ort'),('whisper','managed'))):
            verify(root,files)
            # Wait before creating a worker; a busy workstation never weakens its guard.
            preflight=dict(family=family,engine=engine,started=time.time(),observations=[])
            identity.setdefault('preflights',[]).append(preflight)
            preflight_start=time.monotonic()
            while True:
                available=psutil.virtual_memory().available
                elapsed=time.monotonic()-preflight_start
                preflight['observations'].append(dict(seconds=elapsed,available=available));save()
                if engine!='managed' or available>=20*1024**3:break
                assert elapsed<600,('Managed memory preflight timed out',available)
                time.sleep(2)
            name=f'{number:02d}-{family}-{engine}'
            command=['dotnet',str(base/'bin/MultilingualReplay.dll')] if engine=='managed' else ['C:/Python313/python.exe','-X','utf8','-B',str(frozen/'native.py')]
            command.extend([str(root),str(base/'manifests'/(family+'.json')),str(out/name)])
            before=accounting.snapshot();write_new(out/(name+'-pre.json'),before)
            row=dict(name=name,family=family,engine=engine,command=command,started=time.time(),code=None,samples=0,members={},peak_rss=0,preflight_available=available)
            start=time.monotonic()
            with (out/(name+'.log')).open('x',encoding='utf-8') as log,(out/(name+'-samples.jsonl')).open('x',encoding='utf-8') as samples:
                parent.cpu_affinity([2])
                try:child=subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                finally:parent.cpu_affinity([0])
                process=psutil.Process(child.pid);row.update(pid=child.pid,birth=process.create_time());identity['runs'].append(row);save()
                try:
                    while child.poll() is None:
                        members=[]
                        for member in [process]+process.children(recursive=True):
                            try:
                                item=dict(pid=member.pid,birth=member.create_time(),rss=member.memory_info().rss,affinity=member.cpu_affinity(),cpu_seconds=sum(member.cpu_times()[:2]))
                                assert item['affinity']==[2],item
                                members.append(item);row['members'][str(member.pid)]=item['birth']
                            except psutil.NoSuchProcess:pass
                        sample=dict(seconds=time.monotonic()-start,members=members,available=psutil.virtual_memory().available)
                        rss=sum(v['rss'] for v in members);row['samples']+=1;row['peak_rss']=max(row['peak_rss'],rss)
                        samples.write(json.dumps(sample)+'\n');samples.flush();save()
                        assert rss<20*1024**3 and sample['seconds']<3600 and sample['available']>=1024**3,'Resource guard'
                        time.sleep(.5)
                finally:
                    if child.poll() is None:
                        assert process.create_time()==row['birth']
                        for member in process.children(recursive=True):member.kill()
                        child.kill()
                    row['code']=child.wait();row['ended']=time.time();row['seconds']=time.monotonic()-start;save()
            after=accounting.snapshot();write_new(out/(name+'-post.json'),after)
            row['accounting']=accounting.foreign_fraction(before,after,parent.pid);save()
            assert row['code']==0,name
            for pid,birth in row['members'].items():assert not psutil.pid_exists(int(pid)) or psutil.Process(int(pid)).create_time()!=birth,(pid,birth)
            result=read(out/name/'result.json');assert result['passed'] and len(result['cases'])==41
            print('Completed',name,'seconds',row['seconds'],'peak',row['peak_rss'],flush=True)
        verify(root,files);identity['complete']=True
    except BaseException:
        identity['error']=traceback.format_exc();raise
    finally:
        identity['ended']=time.time();save();parent.cpu_affinity(prior)


if __name__=='__main__':main()
