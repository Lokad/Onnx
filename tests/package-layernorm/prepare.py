"""Fresh Git archive, exact production equivalence and private consumer setup."""
import argparse,subprocess,tarfile,shutil,xml.etree.ElementTree as ET
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);a=p.parse_args();base=Path(a.artifact).resolve();base.mkdir()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','diff','--exit-code',QUALIFIED+'..'+revision,'--',*PATHS],cwd=ROOT,check=True)
    assert pin(PRIOR/'closed.json')['sha256']==RECEIPT;prior=read(PRIOR/'closed.json');links={}
    for name in ['source-identity.json','windows-audit.json','amd-audit.json','verification.json']:
        assert pin(PRIOR/name)==prior['files'][name];links[str((PRIOR/name).relative_to(ROOT))]=pin(PRIOR/name)
    for name,wanted in prior['reports'].items():assert pin(ROOT/name)==wanted;links[name]=wanted
    links[str((PRIOR/'closed.json').relative_to(ROOT))]=pin(PRIOR/'closed.json')
    archive=base/'source.tar';subprocess.run(['git','archive','--format=tar','--output='+str(archive),revision],cwd=ROOT,check=True)
    tree={}
    for row in subprocess.check_output(['git','ls-tree','-rz',revision],cwd=ROOT).split(b'\0'):
        if not row:continue
        meta,name=row.split(b'\t',1);mode,kind,oid=meta.split();assert kind==b'blob';tree[name.decode()]=oid.decode()
    source=base/'source'
    with tarfile.open(archive) as tar:
        members=tar.getmembers();assert len({m.name for m in members})==len(members)
        assert all((m.isfile() or m.isdir()) and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in members)
        assert {m.name for m in members if m.isfile()}==set(tree);tar.extractall(source,filter='data')
    data=subprocess.run(['git','cat-file','--batch'],input=('\n'.join(tree.values())+'\n').encode(),cwd=ROOT,stdout=subprocess.PIPE,check=True).stdout
    cursor=0;inventory={}
    for name,oid in tree.items():
        end=data.index(b'\n',cursor);header=data[cursor:end].split();assert header[:2]==[oid.encode(),b'blob']
        size=int(header[2]);blob=data[end+1:end+1+size];cursor=end+2+size;path=source/name;actual=path.read_bytes();normalized=actual!=blob
        if normalized:assert b'\0' not in blob and actual.replace(b'\r\n',b'\n')==blob,name
        inventory[name]=dict(git_blob=oid,crlf_normalized=normalized,**pin(path))
    assert cursor==len(data);write(base/'source-manifest.json',inventory)
    app=base/'consumer';app.mkdir();(base/'feed').mkdir()
    for name in ['Consumer.csproj','Program.cs']:shutil.copyfile(source/'tests/package-layernorm'/name,app/name)
    config=ET.Element('configuration');sources=ET.SubElement(config,'packageSources');ET.SubElement(sources,'clear')
    ET.SubElement(sources,'add',dict(key='local',value=str(base/'feed')));ET.SubElement(sources,'add',dict(key='nuget.org',value='https://api.nuget.org/v3/index.json'))
    ET.ElementTree(config).write(app/'nuget.config',encoding='utf-8',xml_declaration=True)
    prepared=dict(source_revision=revision,qualified_source=QUALIFIED,equivalent_paths=PATHS,archive=pin(archive),source_manifest=pin(base/'source-manifest.json'),
        product_evidence=links,consumer_files={name:pin(app/name) for name in ['Consumer.csproj','Program.cs','nuget.config']},settings=SETTINGS,
        tools={str(p.relative_to(ROOT)):pin(p) for p in Path(__file__).parent.iterdir() if p.is_file()})
    write(base/'prepared.json',prepared);print(json.dumps(dict(source=revision,files=len(inventory),crlf=sum(v['crlf_normalized'] for v in inventory.values()),prepared=pin(base/'prepared.json'))))

if __name__=='__main__':main()
