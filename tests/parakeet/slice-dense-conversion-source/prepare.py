"""Stage one public override using the existing bounded slice-copy helper."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-slice-dense-conversion-source-20260924'
RELEASE=ROOT/'artifacts/parakeet-observed-dense-where-root-amd-v2-20260924'
COMPONENT=ROOT/'artifacts/parakeet-positional-copy-cost-amd-20260924'
CHANGED='src/Lokad.Onnx/TensorSlice.cs'
TEST='tests/Lokad.Onnx.Tensors.Tests/SliceDenseConversionTests.cs'


def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def read(p):return json.loads(p.read_text(encoding='utf8'))


def main():
    assert not BASE.exists()
    assert pin(RELEASE/'closed.json')['sha256']=='dd612e81a85f74aebe4371ca93e6f17216c6779b927caf66bf400f0710af0f8e'
    release=read(RELEASE/'closed.json');assert release['passed'] and release['analysis']==pin(RELEASE/'analysis.json')
    for name,wanted in release['files'].items():assert pin(RELEASE/name)==wanted,name
    assert pin(COMPONENT/'closed.json')['sha256']=='c4c9aba12446a6216f35e213a5fcd1b3282d1a6261caab0f94029be81af1f8ef'
    component=read(COMPONENT/'closed.json')
    assert component['evidence_qualified'] and not component['component_stable'] and not component['useful_component_estimate']
    review=ROOT/'tests/parakeet/positional-copy-cost-results/observations-20260924.json'
    assert pin(review)['sha256']=='640d00690215f94f782369220834dedb8c40f254cbcc50d9aedf80be5a5fa771'
    decision=read(review);assert decision['original_verdict_unchanged'] and not decision['original_comparison_admitted']
    source={n[7:]:v for n,v in read(RELEASE/'payload.json')['files'].items() if n.startswith('source/')}
    assert len(source)==427 and TEST not in source
    for name,wanted in source.items():assert pin(ROOT/name)==wanted,name
    before=(ROOT/CHANGED).read_text(encoding='utf8')
    needle='    public override Tensor<T> Clone() => ToDenseTensor();'
    assert before.count(needle)==1 and 'public override DenseTensor<T> ToDenseTensor()' not in before
    addition='''    public override DenseTensor<T> ToDenseTensor()
    {
        if (TryCopyContiguousSlice(out var dense)) return dense;
        return base.ToDenseTensor();
    }

'''
    after=before.replace(needle,addition+needle)
    assert after.replace(addition,'')==before
    BASE.mkdir();folder=BASE/'source';folder.mkdir()
    for name in source:
        target=folder/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    (folder/CHANGED).write_text(after,encoding='utf8',newline='\n')
    shutil.copy2(TOOLS/'SliceDenseConversionTests.cs.txt',folder/TEST)
    snapshot={name:pin(folder/name) for name in [*source,TEST]}
    assert [n for n,v in snapshot.items() if source.get(n)!=v]==[CHANGED,TEST]
    patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=CHANGED,tofile=CHANGED))
    (BASE/'candidate.patch').write_text(patch,encoding='utf8')
    shutil.copy2(ROOT/'.agent/m73-parakeet-slice-dense-conversion-20260924.md',BASE/'prospective-plan.md')
    receipt=dict(passed=True,built=False,root_product_changed=False,before=source,source=snapshot,
        changed=[CHANGED,TEST],root_release=pin(RELEASE/'closed.json'),component=pin(COMPONENT/'closed.json'),
        component_comparison_admitted=False,prototype_decision=pin(review),generator=pin(Path(__file__)),
        tests=pin(TOOLS/'SliceDenseConversionTests.cs.txt'),patch=pin(BASE/'candidate.patch'),plan=pin(BASE/'prospective-plan.md'),
        scope='One added override; existing helper, all original method bodies, guards and arithmetic stay unchanged. No performance admission.')
    with (BASE/'prepared.json').open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2);f.write('\n')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),source_files=len(snapshot),changed=receipt['changed'],root_product_changed=False)))


if __name__=='__main__':main()
