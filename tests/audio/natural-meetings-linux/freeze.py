"""Freeze Linux native evidence only after a successful input-only worker."""
from pathlib import Path
import argparse
import json
import re
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--source-commit',required=True);a=p.parse_args();base=a.artifact.resolve()
    assert re.fullmatch('[0-9a-f]{40}',a.source_commit) and not (base/'frozen.json').exists()
    sys.path.insert(0,str(base/'runtime'))
    from common import pin,read,write
    manifest=read(base/'manifest.json')
    sys.path[:0]=manifest['python_path']
    import psutil
    assert read(base/'preparation.json')['passed'] is True
    dependency=read(base/'dependency-check.json');assert dependency['passed'] is True
    assert manifest['native_files']==dependency['native_files'] and manifest['dependency_check']==pin(base/'dependency-check.json')
    for name,wanted in manifest['native_files'].items():assert pin(Path(name))==wanted,name
    directory=base/'process-native-whisper-inputs';state=read(directory/'identity.json');inputs=read(directory/'worker/inputs.json')
    assert state['complete'] is True and state['code']==0 and read(directory/'complete.json')==dict(code=0)
    assert inputs==dict(passed=True,family='whisper',affinity=4,cases=[dict(name=c['name'],samples=c['samples'],pcm_sha256=c['pcm_sha256']) for c in manifest['cases']])
    births={int(pid):birth for pid,birth in state['members'].items()}
    for item in [state['supervisor'],state['child']]:births[item['pid']]=item['birth']
    for pid,birth in births.items():
        try:assert psutil.Process(pid).create_time()!=birth,(pid,birth)
        except psutil.NoSuchProcess:pass
    files={}
    for name in ['runtime','source','reference-source','upstream']:
        for path in (base/name).rglob('*'):
            if path.is_file():files[path.relative_to(base).as_posix()]=pin(path)
    for name in ['manifest.json','preparation.json','dependency-check.json','torch-install.json','dependencies-install.json','prospective-plan.md','predecessor.json']:
        files[name]=pin(base/name)
    write(base/'frozen.json',dict(schema=1,source_commit=a.source_commit,files=files,native_files=manifest['native_files'],
        limits=manifest['limits'],input_smoke=dict(state=pin(directory/'identity.json'),inputs=pin(directory/'worker/inputs.json'),terminal_processes=[dict(pid=p,birth=b) for p,b in births.items()]),
        schedule=[dict(engine='native',family='whisper',cases=[c['name'] for c in manifest['cases']])],
        scope='One independent native Linux recording sequence; retain failed Windows attempt and refused Windows retry; no managed or Parakeet rerun'))
    print('Frozen',len(files),'Linux reference payload files; all input-smoke births terminal.')


if __name__=='__main__':main()
