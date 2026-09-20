"""Freeze complete corpus and promoted graph only after primitive tests pass."""
import argparse,collections,subprocess,shutil,sys
import onnx
from common import *
from promote import build,partition

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',required=True);args=parser.parse_args();base=Path(args.artifact).resolve();base.mkdir()
    assert shutil.disk_usage(base).free>=LIMITS['disk'];files={};verified=0
    for prior,wanted in [(CROSS,CROSS_RECEIPT),(TRACE,'7778c67fa57d567ffa7d779b0960b78027f57fe1dff6bd448f7eca1c74542899')]:
        assert pin(prior/'closed.json')['sha256']==wanted;receipt=read(prior/'closed.json')
        assert receipt.get('passed',receipt.get('structural_passed')) is True
        for name,identity in receipt['files'].items():assert pin(prior/name)==identity,name;verified+=1
        report=receipt['report'];assert pin(ROOT/report['file'])=={k:report[k] for k in ['bytes','sha256']};verified+=1
        files[report['file']]=pin(ROOT/report['file'])
        files[rel(prior/'closed.json')]=pin(prior/'closed.json');files[rel(prior/'manifest.json')]=pin(prior/'manifest.json')
        if 'births' in receipt:assert all(absent(v) for v in receipt['births'])
    cross=read(CROSS/'manifest.json');traced=read(TRACE/'manifest.json');assert len(cross['requests'])==21 and len(traced['outputs'])==41
    trace_path=ROOT/traced['model'];assert pin(MODEL)['sha256']==ORIGINAL and pin(DATA)['sha256']==WEIGHTS
    for path in [MODEL,DATA,trace_path]:files[rel(path)]=pin(path)
    original=onnx.load(MODEL,load_external_data=False);trace=onnx.load(trace_path,load_external_data=False)
    assert set(n.op_type for n in original.graph.node)==OPS and len(original.graph.node)==1559
    promoted=base/'promoted';promoted.mkdir();ledger=build(original,trace,MODEL.parent,promoted)
    stages=partition(onnx.load(promoted/'model.onnx',load_external_data=False),promoted)
    assert len(ledger['promotions'])==753 and len(ledger['lowered_convolutions'])==2 and ledger['nodes']==1571
    assert len(stages)==69 and sum(s['kind']=='math_erf' for s in stages)==34
    write(promoted/'conversion.json',ledger);write(promoted/'stages.json',stages)
    for path in promoted.iterdir():files[rel(path)]=pin(path)
    jobs=[];requests=[]
    for index,item in enumerate(cross['requests']):
        assert item['request']==index;baselines={}
        for engine in ['managed','native']:
            folder=CROSS/'outputs'/f"{engine}-{index:02}-{item['name']}";result=read(folder/'result.json');assert result['complete']
            files[rel(folder/'result.json')]=pin(folder/'result.json')
            for row in result['records']:
                path=folder/row['file'];identity=pin(path);assert identity['sha256']==row['sha256'] and row['shape']==[1,1500,1280]
                files[rel(path)]=identity;baselines[row['kind']]=dict(file=rel(path),format='f32',shape=row['shape'],raw_sha256=row['sha256'])
        assert set(baselines)=={'MM','MN','NM','NN'}
        requests.append(dict(request=index,name=item['name'],baselines=baselines))
        for features in ['managed','native']:
            desc=item[features+'_features'];source(desc);files[desc['file']]=pin(ROOT/desc['file'])
            for engine in ['numpy','ort']:
                jobs.append(dict(id=f"{index:02}-{features}-{engine}",request=index,name=item['name'],features=features,engine=engine,input=desc))
    assert requests[0]['name']==requests[-1]['name'] and len(jobs)==84
    for path in Path(__file__).parent.iterdir():
        if path.is_file():files[rel(path)]=pin(path)
    for path in (ROOT/'artifacts/whisper-full-reference-tooling-20260920').iterdir():
        if path.is_file():files[rel(path)]=pin(path)
    packages();import scipy,scipy.special,onnxruntime,psutil
    assert scipy.__version__=='1.16.3' and onnxruntime.__version__=='1.29.0' and np.__version__=='2.2.4'
    numerical={str(Path(sys.executable)):pin(sys.executable)}
    for package in [np,scipy,onnxruntime,onnx,psutil]:
        root=Path(package.__file__).parent;numerical[str(Path(package.__file__))]=pin(package.__file__)
        for p in root.rglob('*'):
            if p.is_file() and p.suffix.lower() in ['.dll','.pyd']:numerical[str(p)]=pin(p)
        libs=root.parent/(root.name+'.libs')
        if libs.is_dir():
            for p in libs.glob('*.dll'):numerical[str(p)]=pin(p)
    spec=dict(schema=1,protocol='whisper-full-fp64-v1',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        files=files,numerical_files=numerical,original=rel(MODEL),trace_model=rel(trace_path),promoted=rel(promoted),outputs=traced['outputs'],
        jobs=jobs,requests=requests,limits=LIMITS,thread_environment=THREADS,prior_files_verified=verified,
        reference_limit=1e-9,original_limit=1e-4,original_core=cross['core_sha256'],reference_bytes=84*sum(int(np.prod(o['shape']))*8 for o in traced['outputs']))
    assert spec['reference_bytes']==55480320000
    write(base/'manifest.json',spec);print(json.dumps(dict(jobs=84,arrays=3444,reference_bytes=spec['reference_bytes'],manifest=pin(base/'manifest.json'))))

if __name__=='__main__':main()
