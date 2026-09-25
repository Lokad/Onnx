"""Freeze the isolated candidate build, complete IL review and focused graph tests."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('owned_transport',TOOLS.parent/'weight-ownership-probe/run.py')
prior=importlib.util.module_from_spec(loader);loader.loader.exec_module(prior)
pin,read,write,ssh=prior.pin,prior.read,prior.write,prior.ssh
BASE=ROOT/'artifacts/parakeet-owned-packed-weight-scope-recovery-amd-20260925'
FAILED=ROOT/'artifacts/parakeet-owned-packed-weight-scope-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-owned-packed-weight-scope-recovery-20260925'
PRELUDE=prior.PRELUDE.replace(prior.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport=prior.transport;transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE
SOURCE=ROOT/'artifacts/parakeet-owned-packed-weight-scope-source-20260925'
ORIGINAL=ROOT/'artifacts/parakeet-owned-packed-weight-amd-20260925'
RUNTIME=ORIGINAL/'build-collected/runtime'
REMOTE_RUNTIME='/dev/shm/lokad-parakeet-owned-packed-weight-20260925/runtime'


def source_verified():
    value=read(SOURCE/'prepared.json');prior.references()
    assert pin(SOURCE/'prepared.json')['sha256']=='0d6b78df6e7d03863b15c42af0b5e5bff564ffdf89ef5bc77ef34307f6e1e3a1'
    assert value['passed'] and not value['built'] and not value['root_product_changed'] and not value['release_admitted']
    assert value['preparer']==pin(TOOLS.parent/'owned-packed-weight-scope-source/prepare.py')
    assert value['original_source']==pin(ROOT/'artifacts/parakeet-owned-packed-weight-source-20260925/prepared.json')
    for name,wanted in value['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    assert value['plan']==pin(SOURCE/'prospective-plan.md') and value['patch']==pin(SOURCE/'candidate.patch')
    review=read(ORIGINAL/'build-review.json')
    assert review['passed'] and value['original_compiled_review']==pin(ORIGINAL/'build-review.json')
    assert review['product']==value['product']
    raw=ORIGINAL/'build-collected/logs/instructions.json';assert pin(raw)==review['inventory']
    inventory=read(raw);assert inventory['inventory_complete']
    rows=[]
    for row in inventory['observations']:
        methods={k:v for k,v in row['normalized_methods'].items() if k not in row['removed']}
        methods.update(row['candidate_methods'])
        assert set(methods)==set(row['method_flags_after'])
        rows.append(dict(assembly=row['assembly'],before_sha256=row['after_sha256'],normalized_methods=methods,
            method_flags_before=row['method_flags_after'],public_surface=row['public_surface_after']))
    assert [len(r['normalized_methods']) for r in rows]==[3277,697]
    value['inventory']=dict(observations=rows)
    for folder,key in [('parakeet-owned-packed-weight-selection-v2-amd-20260925','diagnosis'),
                       ('parakeet-owned-packed-weight-tests-amd-20260925','corrected_contracts')]:
        path=ROOT/'artifacts'/folder
        assert pin(path/'closed.json')==value[key] and read(path/'closed.json')['passed']
    assert value['core_changed_methods']==['PrepareOwnedMatMulWeights'] and not value['data_changed_methods']
    return value


def prepare():
    assert set(p.name for p in BASE.iterdir())=={'products-observed.json'}
    source=source_verified();assert (TOOLS/'review.py').exists()
    products=read(BASE/'products-observed.json');assert products['passed']
    receipt=read(FAILED/'build-collected/build-collection.json');assert receipt['terminal'] and receipt['code']==1
    for name,wanted in receipt['files'].items():assert pin(FAILED/'build-collected'/name)==wanted,name
    output=(FAILED/'build-collected/logs/backend-build.stdout').read_text()
    assert output.count('error CS1612:')==2 and 'List<Node>.this[int]' in output

    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(encoding='utf8'),str(p))
    bundle=BASE/'bundle';bundle.mkdir()
    def put(name,content):
        p=bundle/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as stream:stream.write(content if isinstance(content,bytes) else content.encode())
    for name in source['source']:
        content=(SOURCE/'source'/name).read_bytes()
        if name=='tests/Lokad.Onnx.Backend.Tests/OwnedPackedWeightTests.cs':
            before=b'graph.Nodes[0].Name = "/pre_encode/out/MatMul";'
            after=b'var node = graph.Nodes[0]; node.Name = "/pre_encode/out/MatMul"; graph.Nodes[0] = node;'
            assert content.count(before)==1;content=content.replace(before,after)
        if name=='tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj':
            for project in ['Lokad.Onnx','Lokad.Onnx.Data']:
                reference=chr(92).join(['..','..','src',project,project+'.csproj'])
                before=('<ProjectReference Include="'+reference+'" />').encode()
                assert content.count(before)==1,(name,before)
                after=''.join('<Reference Include="'+n.removesuffix('.dll')+'"><HintPath>'+p+'</HintPath></Reference>' for n,p in products['sources'].items()).encode() if project=='Lokad.Onnx' else b''
                content=content.replace(before,after)
        put('source/'+name,content)
    put('source/tests/Lokad.Onnx.Backend.Tests/OwnedPackedRuntimeIdentityTests.cs',(TOOLS/'RuntimeIdentityTests.cs.txt').read_bytes())
    put('bridge-source/Program.cs',(TOOLS/'Bridge.cs.txt').read_bytes())
    put('bridge-source/Bridge.csproj',(TOOLS.parent/'selected-profile-build-amd/Bridge.csproj').read_bytes())
    put('bridge-source/global.json',(ROOT/'global.json').read_bytes())
    put('common.py',(TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    put('remote.py',(TOOLS/'vm.py').read_bytes())
    put('plan.md',(ROOT/'.agent/m76-parakeet-owned-packed-weights-20260925.md').read_bytes())
    put('baseline.json',json.dumps(source['inventory'],indent=2))
    runtime={n+'.dll':pin(RUNTIME/(n+'.dll')) for n in ['Lokad.Onnx','Lokad.Onnx.Data','Google.Protobuf','FastBertTokenizer','Lokad.Tokenizers','SixLabors.ImageSharp']}
    assert len(runtime)==6 and all(runtime[n]==v for n,v in source['product'].items())
    spec=dict(boot=1789634288.0,prior=REMOTE_RUNTIME,before_product=source['product'],
        external={**{REMOTE_RUNTIME+'/'+name:value for name,value in runtime.items()},**{p:products['runtime'][n] for n,p in products['sources'].items()}},
        reused_dependencies=products['runtime'],reused_product={n:products['runtime'][n] for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']},
        failed_build_collection=pin(FAILED/'build-collected/build-collection.json'),products_observed=pin(BASE/'products-observed.json'),
        source_prepared=pin(SOURCE/'prepared.json'),failed_release_controls=source['failed_release_controls'],
        release_admitted=False,candidate_selected=True,source_files=len(source['source']),
        core_changed_methods=source['core_changed_methods'],data_changed_methods=source['data_changed_methods'],
        feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=300),
        minimum_free=1024**3,output_limit=512*1024**2,expected_tests={'512':27,'256':27,'scalar':2},
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in bundle.rglob('*'):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    helpers=['weight-ownership-probe/run.py','managed-phase-amd/run.py','managed-phase-amd/remote.py','ort-diagnosis-amd/run.py',
        'feed-forward-cost-diagnostic/isolated_baseline.py','slice-dense-conversion-build-amd/Bridge.cs.txt','selected-profile-build-amd/Bridge.csproj']
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},
        helpers={str((TOOLS.parent/n).relative_to(ROOT)).replace('\\','/'):pin(TOOLS.parent/n) for n in helpers}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),source_files=spec['source_files'])))


def prepared():
    source_verified();value=read(BASE/'prepared.json');spec=read(BASE/'bundle/spec.json')
    assert spec['products_observed']==pin(BASE/'products-observed.json')
    assert spec['failed_build_collection']==pin(FAILED/'build-collected/build-collection.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in value['helpers'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in spec['files'].items():assert pin(BASE/'bundle'/name)==wanted,name


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    else:
        prepared()
        if action=='stage':transport.stage()
        else:
            kind=sys.argv[2];assert kind in ['build','capture']
            if action=='launch':
                if kind=='capture':assert read(BASE/'build-review-transferred.json')['passed']
                transport.launch(kind)
            else:{'observe':prior.observe,'collect':prior.collect}[action](kind)
