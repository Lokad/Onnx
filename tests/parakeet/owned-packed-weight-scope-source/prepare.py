"""Restrict the isolated candidate to the measured feed-forward projections."""
import difflib
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('original_owned_build',TOOLS.parent/'owned-packed-weight-build/run.py')
original=importlib.util.module_from_spec(loader);loader.loader.exec_module(original)
pin,read,write=original.pin,original.read,original.write
BASE=ROOT/'artifacts/parakeet-owned-packed-weight-scope-source-20260925'
DIAGNOSIS=ROOT/'artifacts/parakeet-owned-packed-weight-selection-v2-amd-20260925'
TESTS=ROOT/'artifacts/parakeet-owned-packed-weight-tests-amd-20260925'


def main():
    assert not BASE.exists();before=original.source_verified()
    assert pin(DIAGNOSIS/'closed.json')['sha256']=='fcbde9b4797d0fabfd03a883546296f516996b8f9b44c221fff24458ddebf1af'
    closed=read(DIAGNOSIS/'closed.json');assert closed['passed']
    for name,wanted in closed['files'].items():assert pin(DIAGNOSIS/name)==wanted,name
    diagnosis=read(DIAGNOSIS/'analysis.json')
    assert diagnosis['actual_total_count']==88 and diagnosis['target_count']==87 and diagnosis['extra_count']==1
    assert diagnosis['extra_projection']['name']=='/pre_encode/out/MatMul'
    assert pin(TESTS/'closed.json')['sha256']=='3fc949cad141d2a1fef1e3913436920f87f3cecb92f1bce94892c1037bc080f4'
    tests=read(TESTS/'closed.json');assert tests['passed']
    for name,wanted in tests['files'].items():assert pin(TESTS/name)==wanted,name
    old={name:(original.SOURCE/'source'/name).read_bytes() for name in before['source']}
    values=dict(old);edits=[]
    def replace(name,previous,replacement):
        assert values[name].count(previous.encode())==1,(name,previous)
        changed=values[name].replace(previous.encode(),replacement.encode())
        assert changed.replace(replacement.encode(),previous.encode())==values[name]
        values[name]=changed;edits.append(dict(file=name,before=previous,after=replacement))
    graph='src/Lokad.Onnx/GraphOwnedPacking.cs'
    previous='bool eligible = node.Op == OpType.MatMul && Node.IsStandardDomain(node.Domain) && i == 1 && inputs.Length == 2;'
    replacement='''bool eligible = node.Op == OpType.MatMul && Node.IsStandardDomain(node.Domain) && i == 1 && inputs.Length == 2
                        && node.Name?.Contains("/feed_forward", StringComparison.Ordinal) == true;'''
    replace(graph,previous,replacement)
    test='tests/Lokad.Onnx.Backend.Tests/OwnedPackedWeightTests.cs'
    for previous,replacement in [('copies.TotalCopyBytes','context.LastCopyBytes'),('scratch.TotalScratchBytes','context.LastScratchBytes')]:
        replace(test,previous,replacement)
    assert values[test]==(TESTS/'bundle/source'/test).read_bytes()
    replace(test,'Name = "mm", Op = OpType.MatMul','Name = "/layers.0/feed_forward1/linear2/MatMul", Op = OpType.MatMul')
    marker='    [Fact]\n    public void ExistingPackedRecordAndBudgetRemainUntouched()'
    addition='''    [Fact]
    public void EqualShapePreprocessingProjectionRemainsDense()
    {
        var graph = Graph(4096, 1024);
        graph.Nodes[0].Name = "/pre_encode/out/MatMul";
        var original = graph.Initializers["w"];
        Assert.Equal(0, graph.PrepareOwnedMatMulWeights());
        Assert.Same(original, graph.Initializers["w"]);
        Assert.Equal(0, graph.OwnedPackedWeightCount);
        Assert.Equal(0, graph.OwnedPackedWeightBytes);
        Assert.Empty(graph.PackedWeights);
    }

'''+marker
    replace(test,marker,addition)
    assert len(values)==432 and {n for n in values if values[n]!=old[n]}=={graph,test}
    assert all(values[n]==old[n] for n in values if n not in {graph,test})
    BASE.mkdir();source=BASE/'source';source.mkdir()
    for name,content in values.items():
        path=source/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(content)
    patch=''.join(''.join(difflib.unified_diff(old[n].decode().splitlines(True),values[n].decode().splitlines(True),fromfile=n,tofile=n)) for n in [graph,test])
    (BASE/'candidate.patch').write_text(patch,encoding='utf8',newline='\n')
    (BASE/'prospective-plan.md').write_bytes((ROOT/'.agent/m76-parakeet-owned-packed-weights-20260925.md').read_bytes())
    receipt=dict(passed=True,built=False,release_admitted=False,root_product_changed=False,
        original_source=pin(original.SOURCE/'prepared.json'),original_compiled_review=pin(original.BASE/'build-review.json'),
        diagnosis=pin(DIAGNOSIS/'closed.json'),corrected_contracts=pin(TESTS/'closed.json'),
        before=before['source'],source={n:pin(source/n) for n in values},edits=edits,
        product=diagnosis['product'],failed_release_controls=before['failed_release_controls'],
        core_changed_methods=['PrepareOwnedMatMulWeights'],data_changed_methods=[],selected=before['selected'],
        expected_owned_count=87,expected_owned_bytes=1459617792,excluded_weight='onnx::MatMul_6382',
        arithmetic_methods_unchanged=True,public_surface_unchanged=True,
        tests=dict(normal=26,avx512_disabled=26,hardware_disabled=1),
        plan=pin(BASE/'prospective-plan.md'),patch=pin(BASE/'candidate.patch'),preparer=pin(Path(__file__)))
    write(BASE/'prepared.json',receipt)
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),changed=[graph,test],expected_weights=87,root_product_changed=False)))


if __name__=='__main__':main()
