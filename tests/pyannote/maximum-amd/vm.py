"""Reuse the fixed digest-verified transfer protocol for the pyannote payload."""
from pathlib import Path
import argparse
import importlib.util
import json

path=Path(__file__).with_name('vm_support.py')
if not path.exists():path=Path(__file__).resolve().parents[2]/'whisper/maximum-speech/vm.py'
spec=importlib.util.spec_from_file_location('retained_vm_protocol',path)
protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
protocol.REMOTE='/home/vermorel/Onnx/artifacts/pyannote-maximum-amd-20260919'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('deploy','poll','collect'));parser.add_argument('--artifact',type=Path,required=True)
    args=parser.parse_args();base=args.artifact.resolve()
    if args.action=='deploy':
        previous=base.parent/'whisper-maximum-speech-amd-20260919'
        download=json.loads((previous/'download.json').read_text(encoding='utf-8'))
        collection=previous/'collected/collection.json'
        assert protocol.sha(collection)==download['collection_sha256']
        result=json.loads(collection.read_text(encoding='utf-8'));assert result['complete'] and result['code']==0
    getattr(protocol,args.action)(base)
