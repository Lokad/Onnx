"""Prepare a small immutable source/input payload, without inference."""
from pathlib import Path
import hashlib, json, shutil, subprocess, tarfile

ROOT=Path(__file__).resolve().parents[3]
SOURCE='1d10d22f73282bb00630371887ca3719d2e1553b'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def write(path,value):
    with path.open('x',encoding='utf-8') as stream: json.dump(value,stream,indent=2)


def main():
    base=ROOT/'artifacts/wespeaker-frame-product-amd-20260920';base.mkdir()
    stage=base/'payload';stage.mkdir();source=stage/'source';source.mkdir()
    paths=['src','tests/Lokad.Onnx.Backend.Tests','tests/Shared','tests/pyannote/frame-product',
           'global.json','README.md','icon.png','LICENSE.txt']
    archive=base/'source.tar'
    subprocess.run(['git','archive','--format=tar','--output',str(archive),SOURCE,'--',*paths],cwd=ROOT,check=True)
    with tarfile.open(archive) as tar:tar.extractall(source,filter='data')
    tree=subprocess.check_output(['git','ls-tree','-rl',SOURCE,'--',*paths],cwd=ROOT,text=True).splitlines()
    assert len(tree)==342
    for line in tree:
        metadata,name=line.split('\t',1);mode,kind,blob,size=metadata.split();raw=(source/name).read_bytes()
        assert mode=='100644' and kind=='blob' and len(raw)==int(size)
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==blob,name
    proof=ROOT/'artifacts/wespeaker-precision-20260920'
    assert pin(proof/'closed.json')['sha256']=='16aadd8e4e890c289a68f8bb0dccb738de1d87248396cc046f4b619177c05b0a'
    assert pin(proof/'manifest.json')['sha256']=='a75956d24bde4a7882e4f313d082d2aa0ef976b6a74c554f83c46bde44600600'
    reference=stage/'reference';reference.mkdir();shutil.copyfile(proof/'manifest.json',reference/'manifest.json')
    shutil.copytree(proof/'inputs',reference/'inputs')
    shutil.copyfile(Path(__file__).with_name('remote.py'),stage/'remote.py')
    shutil.copyfile(ROOT/'.agent/m3-filterbank-frame-product-20260920.md',stage/'prospective-plan.md')
    spec=json.loads((reference/'manifest.json').read_text());assert len(spec['cases'])==53
    for case in spec['cases']:assert (reference/'inputs'/(case['name']+'.f32')).stat().st_size==case['samples']*4
    tools=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    subprocess.run(['git','diff','--exit-code',tools,'--',str(Path(__file__).parent)],cwd=ROOT,check=True)
    payload=dict(source=SOURCE,tools=tools,source_archive=pin(archive),source_files=len(tree),
                 files={p.relative_to(stage).as_posix():pin(p) for p in sorted(stage.rglob('*')) if p.is_file()})
    write(stage/'payload.json',payload)
    bundle=base/'payload.tar.gz'
    with tarfile.open(bundle,'w:gz') as tar:
        for p in sorted(stage.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(stage).as_posix(),recursive=False)
    write(base/'preparation.json',dict(payload=pin(stage/'payload.json'),archive=pin(bundle),source=SOURCE,tools=tools))
    print(json.dumps(dict(bundle=pin(bundle),files=len(payload['files']),source_files=len(tree))))


if __name__=='__main__':main()
