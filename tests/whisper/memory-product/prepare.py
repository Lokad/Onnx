"""Archive current projects and overlay only the five already-tested candidate files."""
import difflib,shutil,subprocess,tarfile,xml.etree.ElementTree as ET
from common import *

BASE=ROOT/'artifacts/whisper-memory-product-20260921'
CANDIDATE=ROOT/'artifacts/whisper-weight-sharing-20260920'
CHANGES=['src/Lokad.Onnx/ComputationalGraph.cs','src/Lokad.Onnx.Data/WhisperTranscriber.cs',
    'src/Lokad.Onnx.Data/WhisperDecoderWeights.cs','tests/Lokad.Onnx.Backend.Tests/ReleasedBufferCacheTests.cs',
    'tests/Lokad.Onnx.Backend.Tests/WhisperDecoderWeightsTests.cs']
PATHS=['src','tests/Lokad.Onnx.Backend.Tests','tests/Lokad.Onnx.Tensors.Tests','tests/Lokad.Onnx.Bench',
    'tests/Lokad.Onnx.E5Runner','tests/Lokad.Onnx.OpDump','tests/Shared','Lokad.Onnx.slnx','global.json',
    'README.md','CHANGELOG.md','LICENSE.txt','icon.png','pack.cmd']


def same_content(a,b):
    left=Path(a).read_bytes();right=Path(b).read_bytes()
    return left==right or (b'\0' not in left and b'\0' not in right and left.replace(b'\r\n',b'\n')==right.replace(b'\r\n',b'\n'))


def main():
    assert read(CANDIDATE/'local-final-verification.json')['passed'] and read(CANDIDATE/'built.json')['tests_passed']
    candidate=read(CANDIDATE/'source.json')
    for name,wanted in candidate['files'].items():assert pin(CANDIDATE/'source'/name)==wanted,name
    differences=[n for n in candidate['files'] if not (ROOT/n).exists() or not same_content(ROOT/n,CANDIDATE/'source'/n)]
    assert set(differences)==set(CHANGES),differences
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    BASE.mkdir();archive=BASE/'source.tar'
    subprocess.run(['git','archive','--format=tar','--output='+str(archive),revision,*PATHS],cwd=ROOT,check=True)
    tree={}
    for row in subprocess.check_output(['git','ls-tree','-rz',revision,'--',*PATHS],cwd=ROOT).split(b'\0'):
        if row:
            meta,name=row.split(b'\t',1);mode,kind,oid=meta.split();assert kind==b'blob';tree[name.decode()]=oid.decode()
    source=BASE/'source'
    with tarfile.open(archive) as tar:
        members=tar.getmembers();assert len({m.name for m in members})==len(members)
        assert all((m.isfile() or m.isdir()) and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in members)
        assert {m.name for m in members if m.isfile()}==set(tree);tar.extractall(source,filter='data')
    data=subprocess.run(['git','cat-file','--batch'],input=('\n'.join(tree.values())+'\n').encode(),cwd=ROOT,stdout=subprocess.PIPE,check=True).stdout
    cursor=0;original={}
    for name,oid in tree.items():
        end=data.index(b'\n',cursor);header=data[cursor:end].split();assert header[:2]==[oid.encode(),b'blob']
        size=int(header[2]);blob=data[end+1:end+1+size];cursor=end+2+size;path=source/name;actual=path.read_bytes()
        assert actual==blob or (b'\0' not in blob and actual.replace(b'\r\n',b'\n')==blob),name
        original[name]=dict(git_blob=oid,**pin(path))
    assert cursor==len(data);write(BASE/'original-source-manifest.json',original)
    changes={};patch=[]
    for name in CHANGES:
        target=source/name;before=target.read_text(encoding='utf-8') if target.exists() else ''
        old=pin(target) if target.exists() else None;shutil.copyfile(CANDIDATE/'source'/name,target)
        changes[name]=dict(before=old,after=pin(target))
        patch.extend(difflib.unified_diff(before.splitlines(True),target.read_text(encoding='utf-8').splitlines(True),fromfile='original/'+name,tofile='candidate/'+name))
    (BASE/'candidate.patch').write_text(''.join(patch),encoding='utf-8')
    for name in candidate['files']:assert same_content(source/name,CANDIDATE/'source'/name),name
    inventory={p.relative_to(source).as_posix():pin(p) for p in sorted(source.rglob('*')) if p.is_file()}
    write(BASE/'source-manifest.json',inventory)
    app=BASE/'consumer';app.mkdir();(BASE/'feed').mkdir();folder=Path(__file__).parent
    for name in ['Consumer.csproj','Program.cs']:shutil.copyfile(folder/name,app/name)
    config=ET.Element('configuration');sources=ET.SubElement(config,'packageSources');ET.SubElement(sources,'clear')
    ET.SubElement(sources,'add',dict(key='local',value=str(BASE/'feed')));ET.SubElement(sources,'add',dict(key='nuget.org',value='https://api.nuget.org/v3/index.json'))
    ET.ElementTree(config).write(app/'nuget.config',encoding='utf-8',xml_declaration=True)
    shutil.copyfile(ROOT/'.agent/m5-whisper-memory-product-20260921.md',BASE/'prospective-plan.md')
    proof={p.relative_to(ROOT).as_posix():pin(p) for p in [CANDIDATE/'source.json',CANDIDATE/'built.json',CANDIDATE/'local-closed.json',CANDIDATE/'local-final-verification.json',CANDIDATE/'test-results/backend.trx']}
    tools={p.relative_to(ROOT).as_posix():pin(p) for p in folder.iterdir() if p.is_file()}
    for path in folder.glob('*.py'):compile(path.read_text(encoding='utf-8'),str(path),'exec')
    write(BASE/'prepared.json',dict(source_revision=revision,qualified_source='18e10e3+tested private memory changes',archive=pin(archive),
        source_manifest=pin(BASE/'source-manifest.json'),original_manifest=pin(BASE/'original-source-manifest.json'),changes=changes,
        product_evidence=proof,consumer_files={n:pin(app/n) for n in ['Consumer.csproj','Program.cs','nuget.config']},settings=SETTINGS,tools=tools))
    print(json.dumps(dict(prepared=True,source_files=len(inventory),changes=list(changes),receipt=pin(BASE/'prepared.json'))))


if __name__=='__main__':main()
