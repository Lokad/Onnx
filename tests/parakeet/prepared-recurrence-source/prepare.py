"""Prepare only the bounded decoder cache and its lifecycle/dispatch contracts in isolation."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-prepared-recurrence-source-20260924'
PARENT=ROOT/'artifacts/parakeet-wide-entry-first-use-source-v2-20260923'
CAPTURE=ROOT/'artifacts/parakeet-decoder-lstm-capture-amd-20260924'
PLAN=ROOT/'.agent/m64-parakeet-prepared-recurrence-20260924.md'


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    assert not BASE.exists()
    assert pin(PARENT/'prepared.json')['sha256']=='72c22bee93652ed8d6a759c2a984d965fa341697e869cf7a2461a908727483f6'
    assert pin(CAPTURE/'closed.json')['sha256']=='28c7afe448ed16e3bb19d29232c3f90eb2d72c096196ae27792f5261afa5b64f'
    proof=read(CAPTURE/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(CAPTURE/name)==wanted,name
    analysis=read(CAPTURE/'analysis.json');assert analysis['capture']['calls']==380 and analysis['native']['arrays']==2280
    parent=read(PARENT/'prepared.json');assert parent['passed'] and len(parent['source'])==422
    for name,wanted in parent['source'].items():assert pin(ROOT/name)==wanted==pin(PARENT/'source'/name),name
    review=ROOT/'tests/parakeet/selected-profile-results/decoder-lstm-source-observations-20260924.json'
    assert pin(review)['sha256']=='a3cfb36b3ac99369a3bfc3f92e232fa3656d3307183485ee2ed872a18f03b3db'
    changed={};originals={}
    def edit(name,before,after):
        if name not in changed:
            originals[name]=(PARENT/'source'/name).read_text(encoding='utf8');changed[name]=originals[name]
        assert changed[name].count(before)==1,(name,before)
        changed[name]=changed[name].replace(before,after)
    prefix='src/Lokad.Onnx/'
    file=prefix+'ComputationalGraph.cs'
    edit(file,'    internal Dictionary<float[], PackedConvWeight> PackedConvWeights = new Dictionary<float[], PackedConvWeight>();',
        '    internal Dictionary<float[], PackedConvWeight> PackedConvWeights = new Dictionary<float[], PackedConvWeight>();\n'
        '    internal Dictionary<float[], PackedLstmWeight> PackedLstmWeights = new Dictionary<float[], PackedLstmWeight>();')
    edit(file,'                PackedConvWeights.Clear();','                PackedConvWeights.Clear();\n                PackedLstmWeights.Clear();')
    edit(file,'PackedMatMulWeights = PackedWeights, PackedConvWeights = PackedConvWeights',
        'PackedMatMulWeights = PackedWeights, PackedConvWeights = PackedConvWeights, PackedLstmWeights = PackedLstmWeights')
    edit(file,'        GraphConvPacking.PackWeights(this);','        GraphConvPacking.PackWeights(this);\n        GraphLstmPacking.PackWeights(this);')
    edit(prefix+'GraphExecution.cs','        PackedConvWeights = prepared.PackedConvWeights;',
        '        PackedConvWeights = prepared.PackedConvWeights;\n        PackedLstmWeights = prepared.PackedLstmWeights;')
    edit(prefix+'TensorExecutionOptions.cs','    internal IReadOnlyDictionary<float[], PackedConvWeight>? PackedConvWeights { get; init; }',
        '    internal IReadOnlyDictionary<float[], PackedConvWeight>? PackedConvWeights { get; init; }\n\n'
        '    /// <summary>Immutable bounded LSTM projection weights shared by graph contexts.</summary>\n'
        '    internal IReadOnlyDictionary<float[], PackedLstmWeight>? PackedLstmWeights { get; init; }')
    edit(prefix+'GraphPacking.cs','        long retained = GraphConvPacking.PruneAndBytes(graph);',
        '        long retained = GraphConvPacking.PruneAndBytes(graph);\n'
        '        retained += GraphLstmPacking.PruneAndBytes(graph, retained);')
    file=prefix+'CPUExecutionProvider.Recurrent.cs'
    edit(file,'        int H = hiddenSize;\n        using var projections =',
        '''        int H = hiddenSize;
        float[]? preparedInput = null, preparedRecurrent = null;
        if (seq == 1 && batch == 1 && numDirections == 1 && !reverse && inputSize == 640 && H == 640
            && opts.Tensor.UseSimd && System.Numerics.Vector.IsHardwareAccelerated)
        {
            preparedInput = GraphLstmPacking.Resolve(opts.Tensor.PackedLstmWeights, (Tensor<float>)W);
            preparedRecurrent = GraphLstmPacking.Resolve(opts.Tensor.PackedLstmWeights, (Tensor<float>)R);
        }
        using var projections =''')
    edit(file,'                    if (projections is not null)\n',
        '''                    if (preparedInput is not null && preparedRecurrent is not null)
                    {
                        LstmProjectOrdered(xs.Slice(xOff, inputSize), preparedInput, xw);
                        LstmProjectOrdered(hv, preparedRecurrent, hr);
                    }
                    else if (projections is not null)
''')
    added={prefix+'GraphLstmPacking.cs':'GraphLstmPacking.cs',
           'tests/Lokad.Onnx.Backend.Tests/PreparedLstmWeightsTests.cs':'PreparedLstmWeightsTests.cs'}
    for name,source in added.items():
        assert name not in parent['source'] and not (ROOT/name).exists();changed[name]=(TOOLS/source).read_text(encoding='utf8')
    assert len(changed)==7
    BASE.mkdir()
    for name in parent['source']:
        target=BASE/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PARENT/'source'/name,target)
    for name,value in changed.items():
        with (BASE/'source'/name).open('w',encoding='utf8',newline='\n') as stream:stream.write(value)
    sources={name:pin(BASE/'source'/name) for name in [*parent['source'],*added]}
    differences=sorted(name for name,wanted in sources.items() if parent['source'].get(name)!=wanted)
    assert differences==sorted(changed) and len(sources)==424
    patch=''.join(''.join(difflib.unified_diff(originals.get(name,'').splitlines(True),text.splitlines(True),fromfile=name,tofile=name)) for name,text in changed.items())
    (BASE/'candidate.patch').write_text(patch,encoding='utf8',newline='\n');shutil.copy2(PLAN,BASE/'prospective-plan.md')
    result=dict(passed=True,built=False,root_product_changed=False,parent=pin(PARENT/'prepared.json'),capture=pin(CAPTURE/'closed.json'),
        source_review=pin(review),before=parent['source'],source=sources,changed=differences,
        product_changes=sorted(name for name in changed if name.startswith('src/')),generator=pin(Path(__file__)),
        added_sources={name:dict(path=(TOOLS/source).relative_to(ROOT).as_posix(),**pin(TOOLS/source)) for name,source in added.items()},
        patch=pin(BASE/'candidate.patch'),plan=pin(BASE/'prospective-plan.md'),budgets=dict(encoder=256*1024**2,decoder=64*1024**2),
        scope='Internal bounded W/R preparation, aggregate retention, context/options propagation and one-step H640 dispatch only. The existing ordered projection kernel, gates, all unrelated source and public signatures are unchanged. No product build, candidate inference or speed claim.')
    with (BASE/'prepared.json').open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),source_files=len(sources),changed=differences,built=False)))


if __name__=='__main__':main()
