"""Stage only local files; no remote change or model execution."""
import shutil,subprocess,tarfile
from pathlib import Path
from common import ROOT,BASE,LOCAL,SHARING,PRIOR,REMOTE_SHARING,REMOTE_RECORDING,LIMITS,pin,read,write


def main():
    assert read(LOCAL/'local-final-verification.json')['passed']
    closed=read(LOCAL/'local-closed.json');assert closed['passed']
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    prepared=read(LOCAL/'prepared.json')
    for name,wanted in prepared['files'].items():assert pin(LOCAL/name)==wanted,name
    for name,wanted in prepared['inputs'].items():assert pin(ROOT/name)==wanted,name
    BASE.mkdir();payload=BASE/'payload';payload.mkdir();folder=Path(__file__).parent
    inherited=read(SHARING/'frozen.json');uploads={};links={}
    def add(name,path):
        path=Path(path);target=payload/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target);uploads[name]=pin(target)
    for path in sorted((LOCAL/'bin').iterdir()):
        if path.suffix not in ['.dll','.json']:continue
        name='bin/'+path.name;wanted=pin(path)
        if inherited['files'].get(name)==wanted:links[name]=dict(source=REMOTE_SHARING+'/'+name,pin=wanted)
        else:add(name,path)
    for path in sorted((LOCAL/'source').glob('*.*')):add('source/'+path.name,path)
    for path in sorted(folder.glob('*.py')):
        compile(path.read_text(encoding='utf-8'),str(path),'exec');add('tools/'+path.name,path)
    add('runtime/supervise.py',folder/'supervise.py');add('runtime/common.py',folder/'common.py')
    add('inputs/inputs.json',PRIOR/'inputs/inputs.json')
    for name in ['connected.npy','shifted.npy']:
        links['inputs/'+name]=dict(source=REMOTE_RECORDING+'/inputs/'+name,pin=pin(PRIOR/'inputs'/name))
    short=ROOT/'artifacts/asr-labeled-20260919/native-whisper/manifest.json'
    add('short/manifest.json',short)
    for case in read(short)['cases'][:2]:
        name=case['pcm'];assert Path(name).name==name
        wanted=pin(short.parent/name);source='assets/pcm/'+case['pcm_sha256']+'.npy'
        assert inherited['files'][source]==wanted
        links['short/'+name]=dict(source=REMOTE_SHARING+'/'+source,pin=wanted)
    for name,path in [('native-recording.json',PRIOR/'native-corrected/manifest.json'),('original-managed.json',PRIOR/'managed/result.json'),
        ('windows-result.json',LOCAL/'local/worker/result.json'),('windows-closure.json',LOCAL/'local-closed.json'),
        ('windows-verification.json',LOCAL/'local-final-verification.json'),('windows-prepared.json',LOCAL/'prepared.json'),
        ('weight-census.json',ROOT/'artifacts/whisper-weight-sharing-20260920/weight-census.json')]:add('reference/'+name,path)
    add('prospective-plan.md',ROOT/'.agent/m4-whisper-memory-contracts-20260920.md')
    archive=BASE/'payload.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in uploads:tar.add(payload/name,arcname=name,recursive=False)
    value=dict(prepared=True,source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        local_closure=pin(LOCAL/'local-closed.json'),local_prepared=pin(LOCAL/'prepared.json'),inherited_frozen=pin(SHARING/'frozen.json'),
        archive=pin(archive),uploads=uploads,links=links,limits=LIMITS,completed_requests=13,refusals=16,
        models={('/home/vermorel/Onnx/'+n):v for n,v in prepared['inputs'].items() if n.startswith('models/')})
    write(BASE/'prepared.json',value)
    print(__import__('json').dumps(dict(prepared=True,receipt=pin(BASE/'prepared.json'),uploads=len(uploads),links=len(links))))


if __name__=='__main__':main()
