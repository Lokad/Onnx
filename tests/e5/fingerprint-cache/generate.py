"""Generate an exact fingerprint-transition prototype against the qualified core."""
from pathlib import Path
import argparse,hashlib,json,re,shutil

CORE='187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4'

def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[3];source=root/'src/Lokad.Onnx/ComputationalGraph.cs';base=a.artifact.resolve()
    assert not base.exists();base.mkdir(parents=True)
    text=source.read_text(encoding='utf-8');start=text.index('    long ComputeStructureFingerprint(HashSet<ComputationalGraph>? visiting)')
    body=text[start:text.index('    string? ValidatePreparation()',start)]
    body=body.replace('this','graph')
    for name in ['Nodes','Inputs','Initializers','InputDescs','OutputDescs']:
        body=re.sub(r'(?<![.\w])'+name+r'\b','graph.'+name,body)
    old='''                void MixString(string? v)
                {
                    if (v is null)
                    {
                        MixUlong(0x9E3779B97F4A7C15UL);
                        return;
                    }
                    MixInt(v.Length);
                    foreach (char c in v) MixUlong((ulong)c);
                }'''
    assert body.count(old)==1
    methods=[]
    for name in ['CopyA','CopyB','Train','Cached']:
        signature=f'public static long {name}(ComputationalGraph graph, HashSet<ComputationalGraph>? visiting = null)'
        if name in ('Train','Cached'):signature=f'public static long {name}(ComputationalGraph graph, Dictionary<ComputationalGraph, Entry[]> cache, HashSet<ComputationalGraph>? visiting = null)'
        value=body.replace('long ComputeStructureFingerprint(HashSet<ComputationalGraph>? visiting)',signature)
        value=value.replace('branch.ComputeStructureFingerprint(visiting)',f'{name}(branch, '+('cache, ' if name in ('Train','Cached') else '')+'visiting)')
        if name=='Train':
            value=value.replace('                ulong h =', '                var entries = new List<Entry>();\n                ulong h =',1)
            training=old.replace('                    if (v is null)','                    ulong before = h;\n                    if (v is null)').replace('                        return;','                        entries.Add(new Entry(before, v, h));\n                        return;')
            training=training.replace('                    foreach (char c in v) MixUlong((ulong)c);','                    foreach (char c in v) MixUlong((ulong)c);\n                    entries.Add(new Entry(before, v, h));')
            value=value.replace(old,training).replace('                return (long)h;','                cache[graph] = entries.ToArray();\n                return (long)h;')
        if name=='Cached':
            value=value.replace('                ulong h =','                var entries = cache.GetValueOrDefault(graph) ?? Array.Empty<Entry>();\n                int cursor = 0;\n                ulong h =',1)
            hit='''                    int position = cursor++;
                    if (position < entries.Length)
                    {
                        var entry = entries[position];
                        if (entry.Before == h && string.Equals(entry.Value, v, StringComparison.Ordinal))
                        {
                            h = entry.After;
                            return;
                        }
                    }
'''
            value=value.replace(old,old.replace('                    if (v is null)',hit+'                    if (v is null)',1))
        methods.append(value)
    generated='using Lokad.Onnx;\npublic static class Fingerprints\n{\n    public readonly record struct Entry(ulong Before, string? Value, ulong After);\n'+''.join(methods)+'}\n'
    (base/'Fingerprints.cs').write_text(generated,encoding='utf-8')
    for name in ['Program.cs','Probe.csproj']:shutil.copyfile(Path(__file__).with_name(name),base/name)
    frozen=root/'artifacts/asr-natural-meetings-20260920/bin';assert pin(frozen/'Lokad.Onnx.dll')['sha256']==CORE
    (base/'reference').mkdir()
    for name in ['Lokad.Onnx.dll','Google.Protobuf.dll']:shutil.copyfile(frozen/name,base/'reference'/name)
    shutil.copyfile(source,base/'ComputationalGraph.original.txt')
    value=dict(schema=1,source=pin(source),generator=pin(Path(__file__)),model=pin(root/'models/multilingual-e5-small/model.onnx'),
        files={p.relative_to(base).as_posix():pin(p) for p in base.rglob('*') if p.is_file()},
        scope='Private proof and component timing only; no production changes or neural inference')
    (base/'generation.json').write_text(json.dumps(value,indent=2),encoding='utf-8')
    print('Generated four source-derived traversals; qualified core',CORE)

if __name__=='__main__':main()
