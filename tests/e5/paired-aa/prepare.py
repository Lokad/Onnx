"""Freeze the successfully smoke-checked A/A worker, exact inputs and fixed protocol."""
from pathlib import Path
import argparse,shutil,subprocess,tarfile
from prepare_inputs import CASES,CORE,PROTOBUF,sha,pin,read,write_new

def prepare(base,root):
    assert not (base/'payload').exists() and not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()
    smoke=read(base/'smoke-process.json');audit=read(base/'smoke-audit-final.json')
    assert smoke['complete'] and smoke['code']==0 and audit['passed'] and audit['process_sha256']==sha(base/'smoke-process.json')
    assert audit['auditor_sha256']==sha(Path(__file__).with_name('audit.py'))
    for name,wanted in smoke['binaries'].items():assert pin(base/'bin'/name)==wanted,name
    for name in ('Bridge.cs','Host.cs','Bridge.csproj','Host.csproj'):
        assert pin(Path(__file__).with_name(name))==smoke['source'][name],name
    payload=base/'payload';payload.mkdir();borrowed={}
    def add(source,name):
        target=payload/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    for source in (base/'bin').iterdir():
        if source.suffix in ('.dll','.json'):
            add(source,'bin/'+source.name)
            if source.name in ('Lokad.Onnx.dll','Google.Protobuf.dll'):
                borrowed['bin/'+source.name]='/home/vermorel/Onnx/artifacts/whisper-recording-amd-v2-20260919/bin/'+source.name
    assert sha(payload/'bin/Lokad.Onnx.dll')==CORE and sha(payload/'bin/Google.Protobuf.dll')==PROTOBUF
    for source in (base/'inputs').iterdir():add(source,'inputs/'+source.name)
    for name,wanted in read(base/'inputs/provenance.json')['files'].items():assert pin(payload/'inputs'/name)==wanted
    for source in Path(__file__).parent.iterdir():
        if source.suffix in ('.py','.cs','.csproj','.md'):add(source,source.name)
    add(root/'tests/parakeet/recording-amd/remote.py','process_support.py')
    add(root/'eng/campaign_processes.py','campaign_processes.py')
    add(root/'tests/whisper/maximum-speech/vm.py','vm_support.py')
    add(root/'.agent/m1-paired-aa-20260920.md','plan.md')
    model=root/'models/multilingual-e5-small/model.onnx'
    assert sha(model)=='ca456c06b3a9505ddfd9131408916dd79290368331e7d76bb621f1cba6bc8665'
    schedule=[dict(name=f'v{v}-{CASES[i]}',case=CASES[i],case_index=i,visit=v)
        for v in range(4) for i in (range(5) if v%2==0 else reversed(range(5)))]
    bundle=dict(schema=1,protocol='paired-managed-aa-v1',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        product_source='8732831b52a97b009ab3edbd5319a56269e19449',model=dict(path='/home/vermorel/Onnx/models/multilingual-e5-small/model.onnx',**pin(model)),
        schedule=schedule,borrowed=borrowed,smoke_audit_sha256=sha(base/'smoke-audit-final.json'),
        files={p.relative_to(payload).as_posix():pin(p) for p in sorted(payload.rglob('*')) if p.is_file()})
    write_new(payload/'bundle.json',bundle)
    archive=base/'payload.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for name in list(bundle['files'])+['bundle.json']:
            if name not in borrowed:tar.add(payload/name,arcname=name,recursive=False)
    record=dict(**pin(archive),bundle_sha256=sha(payload/'bundle.json'),logical_bytes=sum(v['bytes'] for v in bundle['files'].values()),
        borrowed_bytes=sum(bundle['files'][name]['bytes'] for name in borrowed))
    write_new(base/'preparation.json',record);print(record)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3])
    a=p.parse_args();prepare(a.artifact.resolve(),a.root.resolve())
