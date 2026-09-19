"""Independent Python Unicode text fixtures for the pinned onnx-asr policy."""
from pathlib import Path
import argparse
import hashlib
import json
import re

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
pieces=['▁Bonjour','▁,','▁été','▁𐐀','▁!','▁','e','\u0301','▁³','▁_','▁😀','<|nospeech|>','▁word','▁123']
cases=[[],[0,1,2,3,4,5],[5,0],[0,5,5,2],[6,7],[5,7],[0,8],[0,9],[0,10],[11],[0,5],[12,13]]
pattern=re.compile(r'\A\s|\s\B|(\s)\b')
rows=[dict(tokens=ids,text=re.sub(pattern,lambda m:' ' if m.group(1) else '', ''.join(pieces[i].replace('▁',' ') for i in ids))) for ids in cases]
result=dict(source='onnx-asr Unicode detokenization policy',reference_revision='675f0e68c24d846ee1775743e92d9b4ed452380e',
    generator_lf_sha256=hashlib.sha256(Path(__file__).read_bytes().replace(b'\r\n',b'\n')).hexdigest(),pieces=pieces,cases=rows)
with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,ensure_ascii=True)
