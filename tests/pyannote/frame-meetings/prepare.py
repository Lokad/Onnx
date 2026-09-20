"""Bind the changed frontend to the unchanged three-request meeting consumer."""
from pathlib import Path
import hashlib, json, shutil, subprocess, tarfile

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-frame-meetings-20260920'


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False)


def main():
    old=ROOT/'artifacts/pyannote-natural-meetings-20260920';product=ROOT/'artifacts/wespeaker-frame-product-amd-v4-20260920'
    assert pin(old/'closed.json')['sha256']=='3ada9904a98aa61ee7fd03db5ce65db206c8278fc13a0939a0ce20249fd5100d'
    assert pin(product/'closed.json')['sha256']=='d02b755dea95632828e1e886fa6ed8b80039cfd4a6454d46de50fcb74297c75a'
    original=read(old/'closed.json');qualified=read(product/'closed.json')
    assert original['closed'] and original['execution_passed'] and original['public_comparison_passed'] and original['all_owned_processes_terminal']
    for name,wanted in original['files'].items():assert pin(old/name)==wanted,name
    for name,wanted in original['sources'].items():assert pin(ROOT/name)==wanted,name
    BASE.mkdir();stage=BASE/'payload';stage.mkdir()
    for name in ['bin','runtime','inputs','prior']:(stage/name).mkdir()
    bindings={}
    def copy(source,target,wanted):
        assert pin(source)==wanted,str(source);shutil.copyfile(source,target)
        assert pin(target)==wanted;bindings[source.relative_to(ROOT).as_posix()]=wanted
    for name in ['NaturalMeetings.dll','NaturalMeetings.deps.json','NaturalMeetings.runtimeconfig.json']:
        copy(old/'bin'/name,stage/'bin'/name,original['files']['bin/'+name])
    for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll','Google.Protobuf.dll','FastBertTokenizer.dll','Lokad.Tokenizers.dll','SixLabors.ImageSharp.dll']:
        source=product/'collected/result/product-bin'/name
        copy(source,stage/'bin'/name,qualified['files'][source.relative_to(ROOT).as_posix()])
    for name in ['Program.cs','NaturalMeetings.csproj','supervise.py','common.py','campaign_processes.py']:
        copy(old/'runtime'/name,stage/'runtime'/name,original['files']['runtime/'+name])
    for name in ['dataset.json','ES2004a-600s.wav','IS1009a-600s.wav']:
        copy(old/'inputs'/name,stage/'inputs'/name,original['files']['inputs/'+name])
    for name,target in [('manifest.json','manifest.json'),('audit.json','scores.json'),
                        ('process-native-run/worker/result.json','native.json'),('process-managed-run/worker/result.json','managed.json')]:
        copy(old/name,stage/'prior'/target,original['files'][name])
    manifest=read(old/'manifest.json')
    manifest.update(core_sha256=pin(stage/'bin/Lokad.Onnx.dll')['sha256'],data_sha256=pin(stage/'bin/Lokad.Onnx.Data.dll')['sha256'],
                    product_source='1d10d22f73282bb00630371887ca3719d2e1553b',
                    accuracy_scope='Changed frontend on the two fixed natural meetings and recovery; retained native outputs; no matched latency claim')
    assert manifest['limits']==dict(rss=8*1024**3,seconds=3600,available=1024**3,preflight=8*1024**3)
    write(stage/'manifest.json',manifest)
    for item in manifest['models'].values():
        assert pin(ROOT/item['path'])=={key:item[key] for key in ['bytes','sha256']}
        bindings[item['path']]=pin(ROOT/item['path'])
    shutil.copyfile(ROOT/'.agent/m3-filterbank-frame-product-20260920.md',stage/'prospective-plan.md')
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    files={p.relative_to(stage).as_posix():pin(p) for p in sorted(stage.rglob('*')) if p.is_file()}
    write(stage/'preparation.json',dict(source=source,product_source=manifest['product_source'],files=files,bindings=bindings,
        prior_receipt=pin(old/'closed.json'),product_receipt=pin(product/'closed.json'),limits=manifest['limits']))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for path in sorted(stage.rglob('*')):
            if path.is_file() and path.suffix!='.wav':tar.add(path,arcname=path.relative_to(stage).as_posix(),recursive=False)
    write(BASE/'transfer.json',dict(archive=pin(BASE/'payload.tar.gz'),preparation=pin(stage/'preparation.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),files=len(files),product_source=manifest['product_source'])))


if __name__=='__main__':main()
