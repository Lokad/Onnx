"""Freeze the exact executable schedule before recognition on either host."""
from pathlib import Path
import argparse
import importlib.metadata
import shutil
import subprocess
import sys
from common import pin,read,write


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[3];base=a.artifact.resolve();source=Path(__file__).parent
    assert not (base/'runtime').exists() and not (base/'frozen.json').exists()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip(), 'Commit executable source before freeze'
    manifest=read(base/'manifest.json');check=read(base/'reference-wrapper-check.json')
    assert check['passed'] is True and check['wrapper']==pin(source/'whisper_recording.py')
    assert check['generator']==pin(base/'reference-source/tests/whisper/recording/generate_reference.py')
    assert read(base/'input-audit.json')['passed'] is True
    assert read(base/'inputs-check.json')['passed'] is True
    validator=read(base/'retained-validator-check.json')
    assert validator['passed'] is True and validator['auditor']==pin(source/'audit.py')
    for name,wanted in read(base/'audit-source-pins.json').items():
        assert pin(base/'audit-source'/name)==wanted,name
    normalizer=read(base/'labels.json')['normalizer']
    assert pin(base/'audit-source/common.py')=={k:normalizer[k] for k in ['bytes','sha256']}
    sys.path.append(str(root/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
    scorer_versions=dict(jiwer='4.0.0',rapidfuzz='3.14.6');scorer_files={}
    for name,version in scorer_versions.items():
        distribution=importlib.metadata.distribution(name);assert distribution.version==version
        directory=Path(distribution.locate_file(name))
        for path in directory.rglob('*'):
            if path.is_file() and path.suffix in ['.py','.pyd','.dll']:scorer_files[path.as_posix()]=pin(path)
    runtime=base/'runtime';runtime.mkdir()
    paths=[p for p in source.iterdir() if p.suffix in ['.py','.cs','.csproj']]
    paths.append(root/'eng/campaign_processes.py')
    for path in paths:
        shutil.copyfile(path,runtime/path.name);assert pin(runtime/path.name)==pin(path)
    shutil.copyfile(root/'.agent/m4-asr-natural-meetings-20260920.md',base/'prospective-plan.md')
    files={'manifest.json':pin(base/'manifest.json'),'prospective-plan.md':pin(base/'prospective-plan.md')}
    for directory in ['bin','runtime','reference-source','upstream','audit-source']:
        for path in sorted((base/directory).rglob('*')):
            if path.is_file():files[path.relative_to(base).as_posix()]=pin(path)
    frozen=dict(schema=1,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        files=files,native_files=manifest['native_files'],labels=pin(base/'labels.json'),input_audit=pin(base/'input-audit.json'),
        wrapper_check=pin(base/'reference-wrapper-check.json'),input_smokes=pin(base/'inputs-check.json'),
        validator_check=pin(base/'retained-validator-check.json'),audit_source_pins=pin(base/'audit-source-pins.json'),
        schedule=[dict(engine=e,family=f,cases=[c['name'] for c in manifest['cases']]) for e in ['native','managed'] for f in manifest['schedule']],
        limits=manifest['limits'],scorer_versions=scorer_versions,scorer_files=scorer_files,
        scope='Twelve new natural-recording accuracy requests; no repeated latency or full intermediate numerical claim')
    write(base/'frozen.json',frozen);print('Frozen',len(files),'payload files,',sum(v['bytes'] for v in files.values()),'bytes.')


if __name__=='__main__':main()
