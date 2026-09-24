"""Freeze one inclusive packing predicate at unchanged selected-release graph budgets."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-inclusive-packing-source-20260924'
PARENT=ROOT/'artifacts/parakeet-wide-entry-first-use-source-v2-20260923'
PROFILE=ROOT/'artifacts/parakeet-selected-profile-amd-20260924'
PLAN=ROOT/'.agent/m63-parakeet-inclusive-packing-20260924.md'
PRODUCT='src/Lokad.Onnx/GraphPacking.cs'
CONTRACT='tests/Lokad.Onnx.Backend.Tests/FoldBudgetTests.cs'
ADDED='tests/Lokad.Onnx.Backend.Tests/PackedBoundaryTests.cs'
TEST_SOURCE=ROOT/'tests/parakeet/packing-admission/PackedBoundaryTests.cs'


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    assert not BASE.exists() and not (ROOT/ADDED).exists()
    assert pin(PARENT/'prepared.json')['sha256']=='72c22bee93652ed8d6a759c2a984d965fa341697e869cf7a2461a908727483f6'
    assert pin(PROFILE/'closed.json')['sha256']=='e6a94afd464807f625e439bdba4a4b71246e641e98b10a632b66bb068adc5937'
    proof=read(PROFILE/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(ROOT/name)==wanted,name
    analysis=read(PROFILE/'analysis.json')
    assert analysis['passed'] and analysis['calls']==240 and analysis['measured_calls']==180
    assert analysis['exact_prior_amd_results'] and len(analysis['diagnostics'])==2
    parent=read(PARENT/'prepared.json');assert parent['passed'] and len(parent['source'])==422
    assert ADDED not in parent['source']
    for name,wanted in parent['source'].items():
        assert pin(PARENT/'source'/name)==wanted and pin(ROOT/name)==wanted,name
    review=ROOT/'tests/parakeet/selected-profile-results/packing-source-observations-20260924.json'
    observations=read(review);assert observations['selected_source']=='81f75c38'
    for name,wanted in observations['local_files'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in observations['ort_files'].items():
        assert pin(ROOT/'artifacts/parakeet-current-source-review-20260923/ort'/name)==wanted,name
    originals={name:(PARENT/'source'/name).read_text(encoding='utf8') for name in [PRODUCT,CONTRACT]}
    product=originals[PRODUCT]
    replacements=[
        ('n < MaxPackedAxis','n <= MaxPackedAxis'),
        ('Upper axis bound of measured packed-kernel territory (P46: 4096, covering GPT-2 c_proj at n=3072; census shows no other model edge in range).',
         'Inclusive reduction-axis bound for prepared packed weights; includes Parakeet projections with 4096 source rows.'),
        ('reduction axis below MaxPackedAxis with total bytes below MaxPackedBytes',
         'reduction axis at most MaxPackedAxis with total bytes at most MaxPackedBytes')]
    for before,after in replacements:
        assert product.count(before)==1;product=product.replace(before,after)
    contract=originals[CONTRACT];before='    [InlineData(4096, 1, false)]'
    assert contract.count(before)==1
    contract=contract.replace(before,'\n'.join([
        '    [InlineData(4096, 1, true)]',
        '    [InlineData(4097, 1, false)]',
        '    [InlineData(4096, 32768, true)]',
        '    [InlineData(4096, 32769, false)]']))
    test=TEST_SOURCE.read_text(encoding='utf8')
    assert 'long budget, int weights)' in test and 'int weights =' not in test
    transformed={PRODUCT:product,CONTRACT:contract,ADDED:test}
    BASE.mkdir()
    for name in parent['source']:
        target=BASE/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(PARENT/'source'/name,target)
    for name,value in transformed.items():
        with (BASE/'source'/name).open('w',encoding='utf8',newline='\n') as stream:stream.write(value)
    sources={name:pin(BASE/'source'/name) for name in [*parent['source'],ADDED]}
    changed=sorted(name for name,value in sources.items() if parent['source'].get(name)!=value)
    assert len(sources)==423 and changed==sorted(transformed)
    assert (BASE/'source'/ADDED).read_text()==TEST_SOURCE.read_text()
    patch=''.join(''.join(difflib.unified_diff(originals.get(name,'').splitlines(True),value.splitlines(True),
        fromfile=name,tofile=name)) for name,value in transformed.items())
    (BASE/'candidate.patch').write_text(patch,encoding='utf8',newline='\n')
    shutil.copy2(PLAN,BASE/'prospective-plan.md')
    result=dict(passed=True,built=False,root_product_changed=False,
        parent=pin(PARENT/'prepared.json'),profile=pin(PROFILE/'closed.json'),source_review=pin(review),
        before=parent['source'],source=sources,changed=changed,product_changes=[PRODUCT],
        generator=pin(Path(__file__)),test_source=pin(TEST_SOURCE),patch=pin(BASE/'candidate.patch'),plan=pin(BASE/'prospective-plan.md'),
        budgets=dict(encoder=256*1024**2,decoder=64*1024**2),
        scope='Only the shared GraphPacking.FitsPackBudget comparison becomes inclusive at reduction length 4096; its comments and boundary tests follow. All arithmetic, other product source, graph budgets and runtime defaults stay selected. No build, inference or performance claim.')
    with (BASE/'prepared.json').open('x',encoding='utf8') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),source_files=len(sources),changed=changed,built=False)))


if __name__=='__main__':main()
