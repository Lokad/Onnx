"""Reuse receipt-bound canonical inputs and complete native outputs, without inference."""
from pathlib import Path
import argparse, hashlib, json, shutil, struct, tarfile

CASES=['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok']
CORE='05884cfd524cc7130321f5dc1bcd0af17dddc7b97e8428d2d2f59e00edb795c2'
PROTOBUF='bf6545a2f705e45724257cee1abe554c2c5dbea5cc4252dfc414fe6d92953cd2'

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def pin(path):return dict(bytes=Path(path).stat().st_size,sha256=sha(path))
def write_new(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def input_hash(inputs):
    value=bytearray(b'LOKAD-CAMPAIGN-INPUTS-1\0')+struct.pack('<i',len(inputs))
    for name,items in sorted(inputs.items()):
        key=name.encode('utf-8');value+=struct.pack('<i',len(key))+key
        value+=struct.pack('<iiiiq',7,2,1,len(items),len(items))
        value+=struct.pack('<'+'q'*len(items),*items)
    return hashlib.sha256(value).hexdigest()

def prepare(root,out):
    assert not out.exists()
    previous=root/'artifacts/e5-public-ort-20260919'
    assert sha(previous/'receipt.json')=='329fae1dccfddbd43d73da14c4c0ff2bbf4e6fca22a8da53ec6ad0231eaf50aa'
    receipt=read(previous/'receipt.json')
    for name in ('provenance.json','results.tar.gz'):
        assert pin(previous/name)==receipt['files'][name],name
    provenance=read(previous/'provenance.json')
    assert sha(root/'models/multilingual-e5-small/model.onnx')==provenance['assets']['model']['sha256']
    prepared=[];out.mkdir(parents=True)
    with tarfile.open(previous/'results.tar.gz','r:gz') as archive:
        for name in CASES:
            member=next(m for m in archive.getmembers() if m.name.endswith('-'+name+'-default.json'))
            stream=archive.extractfile(member);assert stream
            source=json.load(stream);oracle=provenance['oracles'][name]['manifest'];wanted=oracle['outputs'][0]
            assert source['name']==name and source['input_sha256']==input_hash(source['inputs'])==oracle['input_sha256']
            reference=root/'artifacts/e5-current-comparison-20260919/collected/result/model'/(name+'-oracle')/wanted['file']
            assert sha(reference)==wanted['sha256']
            data=reference.read_bytes();count=1
            for size in wanted['dims']:count*=size
            assert len(data)==count*4
            destination=out/(name+'.f32');shutil.copyfile(reference,destination)
            record=dict(name=name,model_sha256=oracle['model_sha256'],input_sha256=source['input_sha256'],
                inputs=source['inputs'],reference_file=destination.name,reference_sha256=wanted['sha256'],shape=wanted['dims'],
                source_archive_member=member.name,source_oracle_sha256=provenance['oracles'][name]['sha256'])
            write_new(out/(name+'.json'),record);prepared.append(record)
    write_new(out/'provenance.json',dict(schema=1,prior_receipt_sha256=sha(previous/'receipt.json'),
        prior_archive_sha256=sha(previous/'results.tar.gz'),cases=CASES,
        files={p.name:pin(p) for p in sorted(out.iterdir()) if p.is_file()}))
    print('Verified and prepared five canonical inputs and complete native outputs.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[3]);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.root.resolve(),a.output.resolve())
