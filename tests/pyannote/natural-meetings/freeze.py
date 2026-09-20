"""Freeze the finite accuracy payload before either engine processes new audio."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import shutil
import subprocess
import sys
from common import CORE, DATA, pin, read, write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[3];base=a.artifact.resolve()
    assert not (base/'frozen.json').exists()
    manifest=read(base/'manifest.json');assert read(base/'input-audit.json')['passed']
    assert pin(base/'bin/Lokad.Onnx.dll')['sha256']==CORE and pin(base/'bin/Lokad.Onnx.Data.dll')['sha256']==DATA
    for engine in ['native','managed']:
        identity=read(base/f'process-{engine}-inputs-v2/identity.json');assert identity['complete'] and identity['code']==0
    assert read(base/'process-native-inputs-v2/worker/inputs.json')==read(base/'process-managed-inputs-v2/worker/inputs.json')
    runtime=base/'runtime';runtime.mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.cs','.csproj','.md'):shutil.copyfile(path,runtime/path.name)
    shutil.copyfile(root/'eng/campaign_processes.py',runtime/'campaign_processes.py')
    shutil.copyfile(root/'tests/pyannote/accuracy/diarization_error.py',runtime/'diarization_error.py')
    shutil.copyfile(root/'.agent/m3-natural-meetings-20260920.md',base/'prospective-plan.md')
    # These are existing files read by the Windows native worker, not remote transfers.
    native_files={}
    for item in list(manifest['models'].values())+list(manifest['native_assets'].values())+list(manifest['upstream'].values())+list(manifest['native_sources'].values()):
        path=root/item['path'];wanted={k:item[k] for k in ['bytes','sha256']};assert pin(path)==wanted
        native_files[path.as_posix()]=wanted
    capi=Path(importlib.metadata.distribution('onnxruntime').locate_file('onnxruntime/capi'))
    binaries=list(capi.glob('*.dll'))+list(capi.glob('*.pyd'))
    assert binaries
    for path in binaries:native_files[path.as_posix()]=pin(path)
    native_files['C:/Python313/python.exe']=pin(Path('C:/Python313/python.exe'))
    # Bind the scorer code and its complete installed package, even though scoring happens after inference.
    metric=root/'artifacts/pyannote-dialogue-20260919/metric-python'
    assert (metric/'pyannote_metrics-4.1.dist-info').is_dir()
    for path in (metric/'pyannote').rglob('*.py'):native_files[path.as_posix()]=pin(path)
    files={}
    for directory in ['inputs','runtime','bin']:
        for path in (base/directory).rglob('*'):
            if path.is_file() and (directory!='bin' or path.suffix in ('.dll','.json')):files[path.relative_to(base).as_posix()]=pin(path)
    for name in ['manifest.json','prospective-plan.md','input-audit.json']:
        files[name]=pin(base/name)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    value=dict(schema=1,source_commit=source,files=files,native_files=native_files,
               schedule=[dict(engine='native',host='Windows',cases=[c['name'] for c in manifest['cases']]),dict(engine='managed',host='AMD',cases=[c['name'] for c in manifest['cases']])],
               prospective_input_failure='Two initial input-only supervisors failed before spawning a worker: psutil required str(path). Both corrected input-only calls pass.',
               limits=manifest['limits'])
    write(base/'frozen.json',value)
    print('Frozen',len(files),'payload files,',sum(v['bytes'] for v in files.values()),'bytes;',len(native_files),'existing native/scorer files; source',source)


if __name__=='__main__':main()
