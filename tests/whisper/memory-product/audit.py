"""Independently verify private package selection, complete outputs and sources."""
import argparse,datetime,re,subprocess,xml.etree.ElementTree as ET,zipfile
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);a=p.parse_args();base=Path(a.artifact).resolve()
    prepared=read(base/'prepared.json');state=read(base/'run.json');assert state['complete'] and state['code']==0 and not state.get('error')
    assert state['prepared']==pin(base/'prepared.json');assert [r['name'] for r in state['runs']]==['solution-build','tensor-tests','pack','restore','build']+[s['name'] for s in SETTINGS]
    assert all(a['ended']<=b['started'] for a,b in zip(state['runs'],state['runs'][1:]));births=[state['supervisor']];resources=[]
    for run in state['runs']:
        assert run['complete'] and run['code']==0 and not run.get('error') and 0<run['seconds']<300
        assert run['members'][str(run['child']['pid'])]==run['child']['birth']
        births += [dict(pid=int(pid),birth=birth) for pid,birth in run['members'].items()]
        samples=[json.loads(v) for v in (base/(run['name']+'.samples.jsonl')).read_text().splitlines()];assert len(samples)==run['samples']>0
        assert all(a['seconds']<=b['seconds'] for a,b in zip(samples,samples[1:]));peak=0
        for row in samples:
            assert 0<=row['seconds']<=run['seconds'] and row['available']>=1024**3
            rss=sum(v['rss'] for v in row['members']);assert rss<4*1024**3;peak=max(peak,rss)
            for member in row['members']:assert member['affinity']==[0] and run['members'][str(member['pid'])]==member['birth']
        resources.append(dict(stage=run['name'],samples=len(samples),seconds=run['seconds'],peak_sampled_group_rss=peak))
    assert all(absent(i) for i in births)
    assert pin(base/'source.tar')==prepared['archive'] and pin(base/'source-manifest.json')==prepared['source_manifest']
    inventory=read(base/'source-manifest.json')
    for name,wanted in inventory.items():assert pin(base/'source'/name)=={k:wanted[k] for k in ['bytes','sha256']},name
    for name,wanted in prepared['consumer_files'].items():assert pin(base/'consumer'/name)==wanted,name
    for name,wanted in prepared['tools'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in prepared['product_evidence'].items():assert pin(ROOT/name)==wanted,name
    prototype=ROOT/'artifacts/whisper-weight-sharing-20260920/source'
    original=read(ROOT/'artifacts/whisper-weight-sharing-20260920/source.json')
    for name,wanted in original['files'].items():
        assert pin(prototype/name)==wanted,name
        left=(base/'source'/name).read_bytes();right=(prototype/name).read_bytes()
        assert left==right or (b'\0' not in left and b'\0' not in right and left.replace(b'\r\n',b'\n')==right.replace(b'\r\n',b'\n')),name
    tests=ET.parse(base/'test-results/tensors.trx').getroot()
    counters=next(e for e in tests.iter() if e.tag.endswith('Counters'))
    assert int(counters.attrib['failed'])==0 and int(counters.attrib['passed'])>0
    budget_rows=[]
    app=base/'consumer';packages=list((base/'source/artifacts/nuget').glob('*.nupkg'));assert len(packages)==1;package=packages[0]
    with zipfile.ZipFile(package) as archive:
        names=archive.namelist();assert len(names)==len(set(names))
        assert {'lib/net10.0/Lokad.Onnx.dll','Lokad.Onnx.nuspec','README.md','LICENSE.txt','CHANGELOG.md','icon.png'}<=set(names)
        assert [n for n in names if n.endswith('.dll')]==['lib/net10.0/Lokad.Onnx.dll']
        spec=ET.fromstring(archive.read('Lokad.Onnx.nuspec'));deps=[(v.attrib['id'],v.attrib['version']) for v in spec.iter() if v.tag.split('}')[-1]=='dependency']
        assert deps==[('Google.Protobuf','3.33.5')]
        assert not re.search(r'\]\((?!https?://|#|mailto:)',archive.read('README.md').decode('utf-8-sig'))
        binary=archive.read('lib/net10.0/Lokad.Onnx.dll');core=dict(bytes=len(binary),sha256=hashlib.sha256(binary).hexdigest())
        for name in ['README.md','CHANGELOG.md','LICENSE.txt','icon.png']:assert archive.read(name)==(base/'source'/name).read_bytes(),name
    cache=base/'packages/lokad.onnx/0.2.0';assert pin(package)==pin(cache/'lokad.onnx.0.2.0.nupkg')==pin(base/'feed/Lokad.Onnx.0.2.0.nupkg')
    assert Path(read(cache/'.nupkg.metadata')['source']).resolve()==base/'feed'
    for path in [cache/'lib/net10.0/Lokad.Onnx.dll',app/'bin/Release/net10.0/Lokad.Onnx.dll',base/'source/src/Lokad.Onnx/bin/Release/net10.0/Lokad.Onnx.dll']:assert pin(path)==core
    assets=read(app/'obj/project.assets.json');assert set(assets['libraries'])=={'Lokad.Onnx/0.2.0','Google.Protobuf/3.33.5'}
    assert Path(assets['project']['restore']['packagesPath']).resolve()==base/'packages'
    assert base/'packages' in [Path(p).resolve() for p in assets['packageFolders']]
    protobuf=pin(base/'packages/google.protobuf/3.33.5/lib/net5.0/Google.Protobuf.dll');assert pin(app/'bin/Release/net10.0/Google.Protobuf.dll')==protobuf
    assert read(base/'packages/google.protobuf/3.33.5/.nupkg.metadata')['source']=='https://api.nuget.org/v3/index.json'
    runtime=read(app/'bin/Release/net10.0/LayerNormPackageConsumer.deps.json');assert {'Lokad.Onnx/0.2.0','Google.Protobuf/3.33.5'}<=set(runtime['libraries'])
    fixture=base/'source/tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx';comparisons=[];baseline=None
    for setting in SETTINGS:
        result=read(base/(setting['name']+'.json'));assert result['complete'] and result['native_loaded'] is False
        assert result['runtime']=='10.0.12' and result['processor_count']==1 and result['affinity']==1
        assert result['flags']==dict(EnableFingerprintStrings=setting['fingerprint'],EnableLayerNormWideOutput=setting['wide'])
        assert Path(result['core']['file']).resolve()==app/'bin/Release/net10.0/Lokad.Onnx.dll' and result['core']['sha256']==core['sha256']
        assert Path(result['protobuf']['file']).resolve()==app/'bin/Release/net10.0/Google.Protobuf.dll' and result['protobuf']['sha256']==protobuf['sha256']
        assert result['fixture_sha256']==pin(fixture)['sha256'] and result['relu']==[0,2]
        mnist=np.asarray(result['mnist'],dtype=np.float32);assert mnist.shape==(3,10) and np.isfinite(mnist).all() and np.array_equal(mnist,mnist[0][None,:].repeat(3,axis=0))
        assert [(v['width'],v['bias']) for v in result['cases']]==[(width,bias) for width in [16,17,384,1280] for bias in [False,True]]
        for case in result['cases']:
            width=case['width'];x=np.asarray(case['input'],dtype=np.float32).astype(np.float64).reshape(3,width)
            scales=np.asarray(case['scales'],dtype=np.float32).astype(np.float64);biases=np.asarray(case['biases'],dtype=np.float32).astype(np.float64)
            centered=x-x.mean(axis=1,keepdims=True);expected=centered/np.sqrt(np.mean(centered**2,axis=1,keepdims=True)+float(np.float32(1e-5)))*scales
            if case['bias']:expected+=biases
            actual=np.asarray(case['actual'],dtype=np.float32).reshape(3,width)
            assert np.isfinite(actual).all() and case['input_unchanged'] and case['held_unchanged']
            assert np.array_equal(actual.ravel().view(np.int32),case['bits'])
            assert np.max(np.abs(np.asarray(case['expected']).reshape(3,width)-expected))<1e-12
            maximum=float((np.abs(actual.astype(np.float64)-expected)/np.maximum(1,np.abs(expected))).max());assert maximum<=1e-6
            comparisons.append(dict(setting=setting['name'],width=width,bias=case['bias'],values=actual.size,max_scaled=maximum))
        assert result['negativeBudgetRefusals']==1 and [r['budget'] for r in result['budgets']]==[0,31,32,64]
        for row in result['budgets']:
            assert row['ownership'] and row['cold']>0 and row['warm']==row['cold']-(32 if row['budget']>=32 else 0)
            assert row['first']==[4*i for i in range(1,9)] and row['second']==[4*i for i in range(20,28)]
        budget_rows.append(dict(setting=setting['name'],rows=result['budgets'],negative_refusals=1))
        selected=dict(relu=result['relu'],mnist=result['mnist'],cases=result['cases'],budgets=result['budgets'])
        if baseline is None:baseline=selected
        else:assert selected==baseline,'Settings changed outputs'
        expected_marker=f"PACKAGE-CONSUMER-PASS fingerprint={int(setting['fingerprint'])} wide={int(setting['wide'])} imports=3 normalizations=8 budgets=4"
        assert (base/(setting['name']+'.stdout')).read_text().strip()==expected_marker
    logs={name:(base/(name+'.stdout')).read_text(encoding='utf-8-sig') for name in ['solution-build','tensor-tests','pack','restore','build']}
    for name,log in logs.items():assert not re.search(r'\berror [A-Z]+\d+\b',log),name
    warnings={name:sorted(set(re.findall(r'^.*\bwarning [A-Z]+\d+.*$',log,re.MULTILINE))) for name,log in logs.items()}
    for run in state['runs']:assert not (base/(run['name']+'.stderr')).read_text().strip(),run['name']
    answer=dict(passed=True,source=prepared['source_revision'],qualified_source=prepared['qualified_source'],source_files=len(inventory),package=pin(package),core=core,
        protobuf=protobuf,births=births,resources=resources,warnings=warnings,comparisons=comparisons,settings=SETTINGS,vector512=read(base/'default.json')['vector512'],
        source_equivalence=True,private_restore=True,prototype_only=True,budgets=budget_rows,tensor_tests=counters.attrib)
    write(base/'audit.json',answer)
    receipt=dict(passed=True,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),audit=pin(base/'audit.json'),births=births,
        files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()})
    write(base/'closed.json',receipt);print(json.dumps(dict(passed=True,files=len(receipt['files']),package=answer['package'],core=core,warnings=warnings,receipt=pin(base/'closed.json'))))

if __name__=='__main__':main()
