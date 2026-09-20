"""Build a new measurement host and explicitly verified 120-file input subset."""
from pathlib import Path
import argparse,hashlib,json,shutil

CASES=['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def pin(p):
    with p.open('rb') as s:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(s,'sha256').hexdigest())
def write(p,v):
    with p.open('x',encoding='utf-8') as s:json.dump(v,s,indent=2)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);a=parser.parse_args()
    root=Path(__file__).resolve().parents[3];base=a.artifact.resolve();base.mkdir(parents=True)
    original=root/'artifacts/gelu-uniform-shortcut-20260920';census=root/'artifacts/gelu-branch-census-20260920'
    assert pin(original/'proof-closed.json')['sha256']=='fda3636d70cb01a7348656a7880947cdaed46e17996811956a5a5ac455f1b727'
    old=read(original/'proof-closed.json')
    for name,wanted in old['files'].items():assert pin(original/name)==wanted,name
    assert pin(census/'closed.json')['sha256']=='9d23fb9f8b38b43ecbdb333707a7259698d0477533a279613d3bfaffc23a650b'
    closed=read(census/'closed.json')
    for name,wanted in closed['files'].items():assert pin(census/name)==wanted,name
    data=base/'payload/data';data.mkdir(parents=True)
    selected={f'capture/{case}/{layer:02d}-{kind}.f32' for case in CASES for layer in range(12) for kind in ['x','bias']}
    for name in sorted(selected):
        target=data/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(census/name,target);assert pin(target)==closed['files'][name]
    shutil.copyfile(census/'closed.json',data/'closed.json')
    generated=base/'generated';generated.mkdir()
    for name in ['Kernels.cs','Probe.csproj']:shutil.copyfile(original/'generated'/name,generated/name)
    program=(original/'generated/Program.cs').read_text(encoding='utf-8')
    start=program.index('foreach(var p in receiptDocument.RootElement.GetProperty("files").EnumerateObject())')
    end=program.index('var variants=',start)
    replacement='''var required=new HashSet<string>(StringComparer.Ordinal);
foreach(string name in new[]{"e5-8tok","e5-30tok","e5-30pad128","e5-128tok","e5-512tok"})
for(int layer=0;layer<12;layer++)foreach(string kind in new[]{"x","bias"})required.Add($"capture/{name}/{layer:D2}-{kind}.f32");
Require(Directory.GetFiles(capture,"*",SearchOption.AllDirectories).Select(f=>Path.GetRelativePath(capture,f).Replace('\\\\','/')).ToHashSet(StringComparer.Ordinal).SetEquals(required.Append("closed.json")),"Input subset coverage");
foreach(string name in required)
{
    var p=receiptDocument.RootElement.GetProperty("files").GetProperty(name);string file=Path.Combine(capture,name);
    Require(new FileInfo(file).Length==p.GetProperty("bytes").GetInt64() && FileHash(file)==p.GetProperty("sha256").GetString(),"Changed subset file "+name);
}
'''
    program=program[:start]+replacement+program[end:]
    program+='\nif(args.Contains("--timing")) Benchmark.Run(capture,output+".timing.json",int.Parse(args[^1]));\n'
    (generated/'Program.cs').write_text(program,encoding='utf-8',newline='\n')
    shutil.copyfile(Path(__file__).with_name('Benchmark.cs'),generated/'Benchmark.cs')
    write(base/'preparation.json',dict(census_receipt=pin(census/'closed.json'),proof_receipt=pin(original/'proof-closed.json'),
        input_subset=True,input_files=len(selected),input_bytes=sum(closed['files'][n]['bytes'] for n in selected),
        files={p.relative_to(base).as_posix():pin(p) for folder in [data,generated] for p in folder.rglob('*') if p.is_file()},
        sources={p.relative_to(root).as_posix():pin(p) for p in Path(__file__).parent.iterdir() if p.is_file()}))
    print('Prepared all 120 required input files and unchanged arithmetic with a new host.')

if __name__=='__main__':main()
