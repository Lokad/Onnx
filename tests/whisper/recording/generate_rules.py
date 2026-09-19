"""Execute pinned upstream Whisper timestamp rules to produce independent mask fixtures."""
from pathlib import Path
import argparse, ast, hashlib, json, types, typing
import numpy as np
import torch

REVISION = '86098128c0b4f24f0e2aa2994de830614b474227'
BEGIN, END, SIZE = 50365, 50257, 51866

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def upstream(source):
    pins=json.loads(Path(__file__).with_name('sources.json').read_text(encoding='utf-8'))
    if sha(source)!=pins['files']['whisper/decoding.py']['sha256']:raise ValueError('Original timestamp source differs')
    tree = ast.parse(source.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'ApplyTimestampRules')
    scope = dict(torch=torch, Tensor=torch.Tensor, np=np, F=torch.nn.functional,
                 Optional=typing.Optional, Tokenizer=object, LogitFilter=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    return scope['ApplyTimestampRules'](types.SimpleNamespace(no_timestamps=BEGIN-1, timestamp_begin=BEGIN, eot=END), 3, 50)

def rows():
    yield 'first-initial-bound', [], -20, {BEGIN+51: 40, BEGIN+50: 10, 32: 50}
    yield 'first-zero', [], -20, {BEGIN: 20}
    yield 'after-opening', [BEGIN], -20, {BEGIN+1: 30, 32: 20}
    yield 'after-text', [BEGIN,32], -20, {33: 20, BEGIN: 30, BEGIN+2: 10}
    yield 'closing-can-repeat', [BEGIN,32,BEGIN+20], -20, {BEGIN+19: 40, BEGIN+20: 30, 33: 50}
    yield 'after-pair-needs-text', [BEGIN,32,BEGIN+20,BEGIN+20], -20, {BEGIN+21: 40, 33: 30}
    yield 'strictly-later-close', [BEGIN,32,BEGIN+20,BEGIN+30,33], -20, {BEGIN+30:40,BEGIN+31:30}
    yield 'closing-can-end', [BEGIN,32,BEGIN+20], -20, {END: 40, BEGIN+20: 30}
    yield 'maximum-close', [BEGIN,32,SIZE-1], -20, {END: 40, SIZE-1:30}
    yield 'maximum-pair', [BEGIN,32,SIZE-1,SIZE-1], -20, {32:40, SIZE-1:50}
    yield 'mass-forces-timestamp', [BEGIN,32], 0, {32:5}
    yield 'text-beats-mass', [BEGIN,32], 0, {32:10}
    yield 'mass-exact-tie', [BEGIN,32], -1000, {32:0,BEGIN+1:0}
    yield 'suppressed-control', [BEGIN,32], -20, {BEGIN-1:100,32:20}
    # Fixed sequences and scores, no output-dependent selection.
    rng = np.random.default_rng(38471)
    prefixes = [[],[BEGIN],[BEGIN,32],[BEGIN,32,BEGIN+150],
                [BEGIN,32,BEGIN+150,BEGIN+151],[BEGIN,32,BEGIN+150,BEGIN+151,33]]
    for i in range(36):
        edits = {int(k):float(v) for k,v in zip(rng.integers(0,SIZE,30),rng.uniform(-12,12,30).astype(np.float32))}
        edits[32] = float(rng.uniform(0,10)); edits[END] = float(rng.uniform(0,10))
        edits[BEGIN+int(rng.integers(1,1500))] = float(rng.uniform(0,10))
        yield f'fixed-{i:02d}', prefixes[i%len(prefixes)], -10, edits

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    assert torch.__version__=='2.11.0+cpu' and np.__version__=='2.2.4'
    rule=upstream(args.source); cases=[]
    for name,tokens,baseline,edits in rows():
        scores=np.full(SIZE,baseline,np.float32)
        for index,value in edits.items(): scores[index]=value
        scores[END+1:BEGIN]=-np.inf
        logits=torch.from_numpy(scores.copy()[None]);prefix=torch.tensor([[50258,50259,50360]+tokens])
        rule.apply(logits,prefix)
        allowed=np.flatnonzero(np.isfinite(logits.numpy()[0]));ranges=[]
        for index in allowed.tolist():
            if ranges and ranges[-1][1]==index:ranges[-1][1]+=1
            else:ranges.append([index,index+1])
        cases.append(dict(name=name,tokens=tokens,baseline=baseline,overrides={str(k):v for k,v in edits.items()},
                          allowed=ranges,argmax=int(logits.argmax())))
    args.output.write_text(json.dumps(dict(revision=REVISION,source_sha256=sha(args.source),torch=torch.__version__,
        numpy=np.__version__,generator_sha256=sha(Path(__file__)),cases=cases),indent=2)+'\n',encoding='utf-8')
    print(len(cases),'independent timestamp masks')

if __name__=='__main__':main()
