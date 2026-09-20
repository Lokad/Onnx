"""Capture all five e5 GELU cases once with the unchanged qualified core."""
from pathlib import Path
import argparse, hashlib, json, os, shutil, subprocess, time, traceback
import psutil

CASES=['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok']
CORE='187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4'
MODEL='ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[3];base=args.artifact.resolve()
    assert base.parent==root/'artifacts' and (base/'bin/Census.dll').exists() and not (base/'frozen.json').exists()
    assert psutil.__version__=='7.0.0'
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip(),'Commit sources before capture'
    previous=root/'artifacts/e5-paired-aa-v2-20260920';closed=read(previous/'closed.json')
    assert pin(previous/'closed.json')['sha256']=='66efc803b7fe177bec64a51e6239114318bbe398244824a3512f939ff072e9e9'
    model=root/'models/multilingual-e5-small/model.onnx'
    assert pin(model)['sha256']==MODEL and pin(base/'bin/Lokad.Onnx.dll')['sha256']==CORE
    inputs=base/'inputs';inputs.mkdir()
    for file in (previous/'inputs').iterdir():
        assert pin(file)==closed['files']['inputs/'+file.name],file
        shutil.copyfile(file,inputs/file.name)
    sources=base/'source';sources.mkdir()
    for file in Path(__file__).parent.iterdir():
        if file.is_file():shutil.copyfile(file,sources/file.name)
    shutil.copyfile(root/'.agent/m2-gelu-branch-census-20260920.md',sources/'plan.md')
    files={p.relative_to(base).as_posix():pin(p) for folder in ['bin','inputs','source'] for p in (base/folder).rglob('*') if p.is_file()}
    frozen=dict(schema=1,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                files=files,model=pin(model),cases=CASES,screen=dict(all_small_fraction=.10,all_large_fraction=.40),
                limits=dict(rss=6*1024**3,seconds=120,available=1024**3),scope='Local input-distribution diagnostic; no performance timing',
                supervisor_source=pin(Path(__file__)),psutil=psutil.__version__)
    write(base/'frozen.json',frozen)
    parent=psutil.Process();old=parent.cpu_affinity();child=None
    identity=dict(supervisor=dict(pid=parent.pid,birth=parent.create_time()),started=time.time(),complete=False,samples=[])
    try:
        assert psutil.virtual_memory().available>=2*1024**3,'Insufficient available memory'
        env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
        with (base/'stdout.txt').open('x',encoding='utf-8') as out,(base/'stderr.txt').open('x',encoding='utf-8') as err:
            parent.cpu_affinity([2])
            try:child=subprocess.Popen(['dotnet',str(base/'bin/Census.dll'),str(model),str(inputs),str(base/'capture')],stdout=out,stderr=err,env=env)
            finally:parent.cpu_affinity([0])
            process=psutil.Process(child.pid);identity['child']=dict(pid=child.pid,birth=process.create_time())
            write(base/'launch.json',dict(supervisor=identity['supervisor'],child=identity['child'],started=identity['started']))
            started=time.monotonic()
            while child.poll() is None:
                try:
                    assert process.create_time()==identity['child']['birth']
                    assert process.cpu_affinity()==[2]
                    sample=dict(seconds=time.monotonic()-started,rss=process.memory_info().rss,available=psutil.virtual_memory().available)
                    identity['samples'].append(sample)
                    assert sample['rss']<=frozen['limits']['rss'] and sample['seconds']<=120 and sample['available']>=1024**3,('Resource guard',sample)
                except psutil.NoSuchProcess:break
                time.sleep(.2)
            identity['exit_code']=child.wait();assert identity['exit_code']==0,'Worker failed; retained stderr'
        for name,wanted in files.items():assert pin(base/name)==wanted,name
        assert pin(model)==frozen['model']
        identity['complete']=True
    except BaseException:
        identity['error']=traceback.format_exc()
        if child is not None and child.poll() is None:
            process=psutil.Process(child.pid);assert process.create_time()==identity['child']['birth']
            child.terminate();child.wait(timeout=15)
        raise
    finally:
        identity['ended']=time.time()
        if child is not None:identity['exit_code']=child.poll()
        write(base/'process.json',identity);parent.cpu_affinity(old)
    print('Captured all five cases and sixty GELU nodes; worker exit',identity['exit_code'])

if __name__=='__main__':main()
