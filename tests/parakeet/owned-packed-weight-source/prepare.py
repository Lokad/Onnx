"""Create one isolated owned-weight candidate from the inspected M73 source."""
import difflib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
sys.path.insert(0,str(TOOLS.parent/'feed-forward-cost-diagnostic'))
from isolated_baseline import qualify,source_path,pin,read
BASE=ROOT/'artifacts/parakeet-owned-packed-weight-source-20260925'
PROOF=ROOT/'artifacts/parakeet-packed-logical-weight-amd-v2-20260925'
OWNERSHIP=ROOT/'artifacts/parakeet-weight-ownership-amd-v2-20260925'


def main():
    assert not BASE.exists();baseline=qualify()
    proof=read(PROOF/'closed.json')
    assert pin(PROOF/'closed.json')['sha256']=='021b990189a2162d9b0b2db991d789434dba177c333b4fa5b97526661e9f8e89'
    assert proof['passed'] and proof['representation_passed']
    for name,wanted in proof['files'].items():assert pin(PROOF/name)==wanted,name
    analysis=read(PROOF/'analysis.json');assert analysis['representation_passed'] and analysis['cases']==48
    costs_path=ROOT/'tests/parakeet/feed-forward-cost-results/costs-20260925.json'
    assert pin(costs_path)['sha256']=='233a927b373399a6c110c59a300c84799873d12e3d71eec6c099a09858d88d06'
    costs=read(costs_path)
    observed=read(OWNERSHIP/'capture-collected/probe/weights.json')
    assert pin(OWNERSHIP/'closed.json')['sha256']=='abe97a41fbb60648a6ee55968cd5e36c861ac908603336bd2f465379cbb1f3cd'
    assert read(OWNERSHIP/'analysis.json')['weights']==pin(OWNERSHIP/'capture-collected/probe/weights.json')
    census={w['name']:w for w in costs['feed_forward_weights']}
    selected=[dict(name=w['Name'],shape=w['Shape'],source_sha256=w['Hash'],bytes=w['Bytes'],consumer=census[w['Name']]['consumers'][0]) for w in observed if not w['Cached']]
    assert len(selected)==87 and sum(w['bytes'] for w in selected)==1459617792
    assert sum(w['shape']==[1024,4096] for w in selected)==39 and sum(w['shape']==[4096,1024] for w in selected)==48
    originals={name:source_path(name).read_bytes() for name in baseline['source_files']}
    values=dict(originals);edits=[]
    def replace(name,before,after):
        content=values[name];newline=b'\r\n' if b'\r\n' in content else b'\n'
        old=before.encode().replace(b'\n',newline);new=after.encode().replace(b'\n',newline)
        assert content.count(old)==1,(name,before)
        changed=content.replace(old,new);assert changed.replace(new,old,1)==content
        values[name]=changed;edits.append(dict(path=name,before=before,after=after))
    name='src/Lokad.Onnx/Global.cs'
    needle='[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("Lokad.Onnx.Backend.Tests")]'
    replace(name,needle,needle+'\n[assembly: System.Runtime.CompilerServices.InternalsVisibleTo("Lokad.Onnx.Data")]')
    name='src/Lokad.Onnx.Data/ParakeetTranscriber.cs'
    needle='        encoder = Load(modelDirectory, "encoder-model.onnx", 256L * 1024 * 1024);'
    replace(name,needle,needle+'\n        encoder.PrepareOwnedMatMulWeights();')
    name='src/Lokad.Onnx/ComputationalGraph.cs'
    needle='        if (typed is BroadcastedTensor<float> broadcast) return TryCollectSingleRoot(broadcast.source, out root);'
    replace(name,needle,'        if (typed is OwnedPackedTensor packed) { root = packed.PackedArray; return true; }\n'+needle)
    needle='        if (typed is BroadcastedTensor<float> broadcast) return CollectAliasRoot(broadcast.source, roots);'
    replace(name,needle,'        if (typed is OwnedPackedTensor packed) { roots.Add(packed.PackedArray); return true; }\n'+needle)
    needle='        if (typed is BroadcastedTensor<float> broadcast) return SharesPooledStorage(candidate, broadcast.source);'
    replace(name,needle,'        if (typed is OwnedPackedTensor packed) return ReferenceEquals(candidate, packed.PackedArray);\n'+needle)
    name='src/Lokad.Onnx/TensorAlias.cs'
    needle='        Memory<T> storage;'
    replace(name,needle,'''        if (t is Tensor<float> typed && OwnedPackedTensor.FindStorage(typed) is { } packed)
        {
            array = packed.PackedArray;
            length = t.Length == 0 ? 0 : packed.PackedArray.Length;
            return true;
        }
'''+needle)
    name='src/Lokad.Onnx/Zzz.WideProjectionEntry.cs'
    needle='        if (ResolvePackedKernel(options, y, m) is { } packedB)'
    replace(name,needle,'        if (TryRunOwnedPacked2D(x, y, destination, options)) return destination;\n\n'+needle)
    name='src/Lokad.Onnx/TensorOps.MatMul.cs'
    needle='    static void RunBatchedFloatMatMul(Tensor<float> bx, Tensor<float> by, Tensor<float> z, TensorExecutionOptions options)\n    {'
    replace(name,needle,needle+'\n        if (TryRunOwnedPackedBatches(bx, by, z, options)) return;')
    name='src/Lokad.Onnx/CPUExecutionProvider.MatMul.cs'
    needle='        if (opts.Optimization != OptimizationMode.Speed) return t;'
    replace(name,needle,needle+'\n        if (t is Tensor<float> owned && OwnedPackedTensor.Resolve(owned) is not null) return t;')
    for template in ['OwnedPackedTensor','GraphOwnedPacking','TensorOps.OwnedPackedMatMul']:
        path='src/Lokad.Onnx/'+template+'.cs';assert path not in values
        values[path]=(TOOLS/(template+'.cs.txt')).read_bytes()
    test='tests/Lokad.Onnx.Backend.Tests/OwnedPackedWeightTests.cs';assert test not in values
    values[test]=(TOOLS/'OwnedPackedWeightTests.cs.txt').read_bytes()
    changed=[name for name in values if values[name]!=originals.get(name)]
    assert len(values)==432 and len(changed)==11
    # Every edited original must reverse byte-for-byte, including line endings.
    reversed_values=dict(values)
    for edit in reversed(edits):
        name=edit['path'];newline=b'\r\n' if b'\r\n' in originals[name] else b'\n'
        old=edit['before'].encode().replace(b'\n',newline);new=edit['after'].encode().replace(b'\n',newline)
        assert reversed_values[name].count(new)==1
        reversed_values[name]=reversed_values[name].replace(new,old)
    assert all(reversed_values[n]==v for n,v in originals.items())
    assert all(values[n]==v for n,v in originals.items() if 'MathOps' in n or n.endswith('Zzz.IsolatedShortMatMul.cs'))
    BASE.mkdir();folder=BASE/'source';folder.mkdir()
    for name,value in values.items():
        path=folder/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(value)
    patch=''.join(''.join(difflib.unified_diff(originals.get(n,b'').decode('utf8').splitlines(True),values[n].decode('utf8').splitlines(True),fromfile=n,tofile=n)) for n in changed)
    (BASE/'candidate.patch').write_text(patch,encoding='utf8',newline='\n')
    (BASE/'prospective-plan.md').write_bytes((ROOT/'.agent/m76-parakeet-owned-packed-weights-20260925.md').read_bytes())
    receipt=dict(passed=True,built=False,root_product_changed=False,release_admitted=False,
        before=baseline['source_files'],source={n:pin(folder/n) for n in values},changed=changed,edits=edits,
        product=baseline['product'],inventory=baseline['inventory'],isolated_evidence=baseline['evidence'],
        failed_release_controls=baseline['failed_release_controls'],representation_closure=pin(PROOF/'closed.json'),
        ownership_closure=pin(OWNERSHIP/'closed.json'),observed_weights=pin(OWNERSHIP/'capture-collected/probe/weights.json'),
        cost_report=pin(costs_path),selected=selected,expected_existing_maps=37,expected_retained_clones=268435456,
        affected_packing_seconds=sum(f['stages']['WeightPacking'] for f in costs['families'] if f['route']=='unmapped'),
        templates={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},patch=pin(BASE/'candidate.patch'),
        plan=pin(BASE/'prospective-plan.md'),core_changed_methods=['TryCollectSingleRoot','CollectAliasRoot','SharesPooledStorage',
            'TryGetTouchedRange','RunWideProjectionMatMul2DCore','RunBatchedFloatMatMul','SpeedDensify'],
        data_changed_methods=['ParakeetTranscriber::.ctor'],arithmetic_methods_unchanged=True,
        tests=dict(normal=25,avx512_disabled=25,hardware_disabled=1))
    with (BASE/'prepared.json').open('x',encoding='utf8') as stream:json.dump(receipt,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),source_files=len(values),changed=changed,selected=len(selected),root_product_changed=False)))


if __name__=='__main__':main()
