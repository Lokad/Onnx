"""Compose the retained dense-mask selector with the fully qualified M66 source."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-observed-dense-where-source-20260924'
PARENT=ROOT/'artifacts/parakeet-validated-composition-source-v2-20260924'
RELEASE=ROOT/'artifacts/parakeet-validated-composition-root-amd-20260924'
APP=ROOT/'artifacts/parakeet-validated-composition-app-amd-20260924'
LAYOUT=ROOT/'artifacts/parakeet-masking-padding-layout-amd-20260924'
OLD=ROOT/'artifacts/parakeet-dense-scalar-where-source-20260924'
NUMERICS=ROOT/'artifacts/parakeet-dense-scalar-where-numerics-amd-20260924'
PLAN=ROOT/'.agent/m70-parakeet-observed-dense-where-20260924.md'
CHANGED='src/Lokad.Onnx/CPUExecutionProvider.Elementwise.cs'
ADDED='src/Lokad.Onnx/Zzz.DenseScalarWhere.cs'


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def closed(folder,digest):
    assert pin(folder/'closed.json')['sha256']==digest
    proof=read(folder/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    return proof


def main():
    assert not BASE.exists()
    closed(RELEASE,'c7a1d2e11566e6eeeb965de6c9cedbf47df479fd51f194c797af412446281609')
    app=closed(APP,'f04c09fbc6c0455c4420d6680d60bb4f8fc5cac9dda2f2507768ba94b86335e4');assert app['admitted']
    closed(NUMERICS,'edd4cf6e9bc9eccb1154df66f9e26419f0c28ebe789962d1d2618d85c1802832')
    assert pin(PARENT/'prepared.json')['sha256']=='af68e6c2c28f5794f3eece39bd9e78fc6809e2ca8925a5616ff2956c405061c4'
    parent=read(PARENT/'prepared.json');assert parent['passed'] and len(parent['source'])==425
    measured=read(RELEASE/'analysis.json')['measured'];application=read(APP/'analysis.json')
    assert measured==application['identities']['candidate']
    controls=[r for r in application['performance']['controls'] if r['role']=='candidate']
    assert len(controls)==21 and all(r['passed'] and r['process_ratio']<=r['limit'] for r in controls)
    for name,wanted in parent['source'].items():assert pin(ROOT/name)==pin(PARENT/'source'/name)==wanted,name

    assert pin(OLD/'prepared.json')['sha256']=='a49c010cfe2f3a47f9b229fb5b3e5af8a83671ca15ff5f2a630d7b4f3f18b3af'
    old=read(OLD/'prepared.json');assert old['passed'] and old['changed']==[CHANGED,ADDED]
    assert pin(ROOT/CHANGED)==old['before'][CHANGED] and ADDED not in parent['source']
    for name in [CHANGED,ADDED]:assert pin(OLD/'source'/name)==old['source'][name],name
    assert (OLD/'source'/ADDED).read_text()==(ROOT/'tests/parakeet/dense-scalar-where-source/DenseScalarWhere.cs').read_text()

    assert pin(LAYOUT/'closed.json')['sha256']=='3884b8ab0d36f8e6f2344bd5a3e7d711c19bebbf6c91bdc402d64d1fe8ac2ac8'
    proof=read(LAYOUT/'closed.json');assert proof['passed']
    for key,name in [('analysis','analysis.json'),('observations','observations.json'),('build_review','build-review.json'),
        ('collection','capture-collected/capture-collection.json'),('audit_correction','audit-correction.json')]:
        assert proof[key]==pin(LAYOUT/name),name
    for name,wanted in read(LAYOUT/'capture-collected/capture-collection.json')['files'].items():
        assert pin(LAYOUT/'capture-collected'/name)==wanted,name
    analysis=read(LAYOUT/'analysis.json')
    assert analysis['passed'] and analysis['core_unchanged'] and analysis['exact_public_results']
    assert (analysis['requests'],analysis['observations'])==(80,9600)
    eligible=0;sizes=[]
    for i in range(80):
        request=read(LAYOUT/'capture-collected/phase'/f'layout-{i:03}.json')
        call,=request['Calls'];assert call['GraphNodes']==2856 and len(call['Records'])==120
        for row in call['Records']:
            if row['Op']!='Where':continue
            c,x,y=row['Inputs'];shape=y['Dimensions'];mask=c['Dimensions']
            assert all(v['ExactDense'] and not v['Reversed'] for v in [c,x,y])
            assert (c['Dtype'],x['Dtype'],y['Dtype'])==('Bool','Float','Float')
            assert y['Length']>=4096 and x['Length']==1 and 1<=len(shape)<=8
            assert len(x['Dimensions'])<=len(shape) and len(mask)<=len(shape)
            offset=len(shape)-len(mask)
            assert all(d==1 or d==shape[offset+a] for a,d in enumerate(mask))
            assert row['TrueCount']==0 and row['FalseCount']==c['Length']
            assert all(v['ArrayBacked'] and v['ArrayOffset']==0 and v['StorageLength']==v['Length'] for v in [c,x,y])
            eligible+=1;sizes.append(y['Length'])
    assert eligible==5760

    BASE.mkdir();source=BASE/'source';source.mkdir()
    for name in parent['source']:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    for name in [CHANGED,ADDED]:shutil.copy2(OLD/'source'/name,source/name)
    after={name:pin(source/name) for name in [*parent['source'],ADDED]}
    assert len(after)==426 and [n for n,w in after.items() if parent['source'].get(n)!=w]==[CHANGED,ADDED]
    patch=''.join(''.join(difflib.unified_diff((ROOT/name).read_text().splitlines(True) if name==CHANGED else [],
        (source/name).read_text().splitlines(True),fromfile=name,tofile=name)) for name in [CHANGED,ADDED])
    (BASE/'candidate.patch').write_text(patch,encoding='utf8');shutil.copy2(PLAN,BASE/'prospective-plan.md')
    value=dict(passed=True,built=False,root_product_changed=False,before=parent['source'],source=after,
        changed=[CHANGED,ADDED],parent=pin(PARENT/'prepared.json'),root_release=pin(RELEASE/'closed.json'),
        application=pin(APP/'closed.json'),measured=measured,existing_application_controls=controls,
        old_source=pin(OLD/'prepared.json'),old_numerics=pin(NUMERICS/'closed.json'),layout=pin(LAYOUT/'closed.json'),
        predicted_uniform_helper_admissions=eligible,observed_output_length_range=[min(sizes),max(sizes)],
        generator=pin(Path(__file__)),plan=pin(BASE/'prospective-plan.md'),patch=pin(BASE/'candidate.patch'),
        scope='Exact retained provider float arm and two-method helper on the qualified current source; no new kernel, threshold or runtime flag. No performance admission.')
    (BASE/'prepared.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),source_files=426,changed=value['changed'],
        predicted_uniform_helper_admissions=eligible,built=False,root_product_changed=False)))


if __name__=='__main__':main()
