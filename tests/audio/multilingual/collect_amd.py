"""Collect closed AMD evidence without changing the closed directory or rerunning writers."""
from pathlib import Path,PurePosixPath,PureWindowsPath
import argparse
import json
import shutil
import subprocess
import tarfile
from common import pin,read,write_new

HOST='vermorel@74.178.91.76'
KEY='C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem'
NAME='asr-multilingual-amd-20260920'


def extract(archive,destination,expected):
    assert not destination.exists(),'Collection destination already exists'
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers();seen=set()
        for member in members:
            name=member.name;parts=PurePosixPath(name).parts
            assert member.isfile() and name and '\\' not in name and ':' not in name
            assert not PurePosixPath(name).is_absolute() and not PureWindowsPath(name).is_absolute()
            assert all(part not in ('.','..') for part in name.split('/')) and name not in seen
            assert name in expected and member.size==expected[name]['bytes'],name
            seen.add(name)
        assert seen==set(expected),'Archive coverage differs from closed receipt'
        destination.mkdir()
        for member in members:
            path=destination/member.name
            assert path.resolve().is_relative_to(destination.resolve())
            path.parent.mkdir(parents=True,exist_ok=True)
            with stream.extractfile(member) as source,path.open('xb') as target:shutil.copyfileobj(source,target)
            assert pin(path)==expected[member.name],member.name
    actual={p.relative_to(destination).as_posix() for p in destination.rglob('*') if p.is_file()}
    assert actual==set(expected)


def remote_archive():
    # Fixed paths, regular files only, and no shell interpolation of data.
    script=r'''
from pathlib import Path
import hashlib,json,sys,tarfile,time
root=Path('/home/vermorel/Onnx');base=root/'artifacts/asr-multilingual-amd-20260920'
assert base.resolve().parent==(root/'artifacts').resolve()
sys.path.insert(0,str(base/'python'));import psutil
def pin(path):
 with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
 return dict(bytes=path.stat().st_size,sha256=digest)
closed=json.loads((base/'closed.json').read_text())
assert closed['closed'] and closed['execution_passed'] and closed['profile']=='amd' and closed['all_owned_processes_terminal']
for item in closed['terminal_processes']:
 try:assert psutil.Process(item['pid']).create_time()!=item['birth'],item
 except psutil.NoSuchProcess:pass
assert {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}==set(closed['files'])|{'closed.json'}
for name,wanted in closed['files'].items():assert pin(base/name)==wanted,name
archive=base.with_name(base.name+'-results.tar.gz');receipt=base.with_name(base.name+'-collection.json')
if receipt.exists():
 value=json.loads(receipt.read_text());assert pin(archive)==value['archive'] and pin(base/'closed.json')==value['closed']
else:
 assert not archive.exists(),'An incomplete archive already exists; preserve it for diagnosis'
 with tarfile.open(archive,'x:gz') as stream:
  for name in sorted(list(closed['files'])+['closed.json']):stream.add(base/name,arcname=name,recursive=False)
 value=dict(schema=1,created=time.time(),archive=pin(archive),closed=pin(base/'closed.json'),all_owned_processes_terminal=True)
 with receipt.open('x') as stream:json.dump(value,stream,indent=2)
print(json.dumps(dict(receipt=value,closed=closed)))
'''
    result=subprocess.run(['ssh','-i',KEY,'-o','BatchMode=yes',HOST,'python3 -B -'],input=script,
                          text=True,encoding='utf-8',capture_output=True,check=True)
    return json.loads(result.stdout)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True)
    args=parser.parse_args();base=args.artifact.resolve();assert base.name==NAME and base.is_dir()
    assert not (base/'collection.json').exists() and not (base/'collected').exists()
    value=remote_archive();expected=value['receipt']['archive'];archive=base/'results.tar.gz'
    if archive.exists():assert pin(archive)==expected
    else:
        number=0
        while (base/f'download-{number}.partial').exists():number+=1
        partial=base/f'download-{number}.partial'
        subprocess.run(['scp','-i',KEY,HOST+':/home/vermorel/Onnx/artifacts/'+NAME+'-results.tar.gz',str(partial)],check=True)
        assert pin(partial)==expected,'Downloaded archive identity differs'
        partial.rename(archive)
    closed=value['closed'];files=dict(closed['files'],**{'closed.json':value['receipt']['closed']})
    extract(archive,base/'collected',files)
    assert read(base/'collected/closed.json')==closed
    root=Path(__file__).resolve().parents[3]
    for name,wanted in closed['sources'].items():assert pin(root/name)==wanted,name
    record=dict(schema=1,passed=True,files=len(files),remote=value['receipt'],archive=pin(archive),
                closed=pin(base/'collected/closed.json'),collector=pin(Path(__file__)))
    write_new(base/'collection.json',record)
    print('Collected and independently verified',len(files),'closed files; receipt',record['closed']['sha256'])


if __name__=='__main__':main()
