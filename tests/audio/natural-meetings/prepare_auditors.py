"""Copy qualified validators and the fixed scorer, or verify an existing copy."""
from pathlib import Path
import argparse
import shutil
from common import pin,read,write


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--verify-existing',action='store_true')
    a=p.parse_args();base=a.artifact.resolve();root=base.parents[1]
    parakeet=root/'artifacts/parakeet-recording-20260919'
    whisper=root/'artifacts/whisper-maximum-speech-20260919'
    sources={
        'parakeet_audit.py':parakeet/'closure-source/tests/parakeet/recording/audit.py',
        'whisper_audit.py':whisper/'reference/recording_audit.py',
        'common.py':root/'tests/audio/multilingual/common.py',
        'scoring.py':root/'tests/audio/multilingual/scoring.py'}
    receipt=read(parakeet/'receipt.json');assert receipt['closed'] is True
    assert pin(sources['parakeet_audit.py'])==receipt['files']['closure-source/tests/parakeet/recording/audit.py']
    receipt=read(whisper/'closed.json');assert receipt['closed'] is True
    assert pin(sources['whisper_audit.py'])==receipt['files']['reference/recording_audit.py']
    wanted={name:pin(path) for name,path in sources.items()}
    if not a.verify_existing:
        (base/'audit-source').mkdir()
        for name,path in sources.items():shutil.copyfile(path,base/'audit-source'/name)
        write(base/'audit-source-pins.json',wanted)
    assert read(base/'audit-source-pins.json')==wanted
    assert {p.name for p in (base/'audit-source').iterdir()}==set(wanted)
    for name,value in wanted.items():assert pin(base/'audit-source'/name)==value,name
    print('Verified four retained validator/scorer sources.')


if __name__=='__main__':main()
