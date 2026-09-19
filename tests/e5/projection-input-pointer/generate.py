"""Change only packed-A address representation; preserve the closed control."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,shutil

def pointer_variant(indexed):
    assert indexed.count('internal static partial class InputPacked')==1
    candidate=indexed.replace('class InputPacked','class InputPointer')
    start=candidate.index('    internal static unsafe void PackedTile12(')
    end=candidate.index('    internal static unsafe void PackedTile8(',start)
    body=candidate[start:end]
    assert body.count('for (int j = 0; j < N; ++j)')==1
    body=body.replace('for (int j = 0; j < N; ++j)','for (int j = 0; j < N; ++j, input += 12)')
    for r in range(12):
        old=f'Vector512.Create(input[j * 12 + {r}])';assert body.count(old)==1
        body=body.replace(old,f'Vector512.Create(input[{r}])')
    return candidate[:start]+body+candidate[end:]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[3];lane=Path(__file__).parent
    assert not a.output.exists()
    old=root/'tests/e5/projection-input-pack/generate.py'
    spec=importlib.util.spec_from_file_location('closed_input_pack_generator',old);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    source=root/'src/Lokad.Onnx/MathOps.PackedAvx512.cs'
    files=g.generate(source.read_text(encoding='utf-8'));files['InputPointer.cs']=pointer_variant(files['InputPacked.cs'])
    a.output.mkdir(parents=True)
    for name,text in files.items():(a.output/name).write_text(text,encoding='utf-8',newline='\n')
    for name in ['Program.cs','Probe.csproj']:shutil.copyfile(lane/name,a.output/name)
    def pin(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    manifest=dict(sources={p.relative_to(root).as_posix():pin(p) for p in [old,source,Path(__file__)]},files={p.name:pin(p) for p in a.output.iterdir()})
    (a.output/'source.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8');print(json.dumps(manifest))

if __name__=='__main__':main()
