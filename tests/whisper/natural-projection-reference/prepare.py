"""Verify prior closed evidence and freeze all 64 projection jobs."""
import argparse,subprocess
import onnx
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);args=p.parse_args()
    base=Path(args.artifact).resolve();base.mkdir()
    receipt=PRIOR/'closed-v2.json';assert pin(receipt)['sha256']==RECEIPT
    old=read(receipt);assert old['passed'] and old['extraction_bridges_passed']
    for name,wanted in old['files'].items():assert pin(PRIOR/name)==wanted,name
    verify(old['reports']);assert all(absent(i) for i in old['births'])
    assert pin(MODEL)['sha256']==MODEL_SHA and pin(WEIGHTS)['sha256']==WEIGHTS_SHA
    graph=onnx.load(MODEL,load_external_data=False).graph
    nodes={n.name:n for n in graph.node};init={i.name:i for i in graph.initializer}
    source=read(PRIOR/'manifest.json');files={rel(receipt):pin(receipt),rel(PRIOR/'manifest.json'):pin(PRIOR/'manifest.json'),rel(MODEL):pin(MODEL),rel(WEIGHTS):pin(WEIGHTS)}
    for name,pj in PROJECTIONS.items():
        node=nodes[pj['node']];assert node.op_type=='MatMul'
        assert list(node.input)==[source['outputs'][pj['input']]['name'],pj['weight']]
        if name=='k':assert list(node.output)==[source['outputs'][pj['output']]['name']]
        else:
            adds=[n for n in graph.node if n.op_type=='Add' and list(n.output)==[source['outputs'][pj['output']]['name']]]
            assert len(adds)==1 and set(adds[0].input)=={node.output[0],pj['bias']['name']}
        for key,desc in [(pj['weight'],pj)]+([] if pj['bias'] is None else [(pj['bias']['name'],pj['bias'])]):
            i=init[key];ext={v.key:v.value for v in i.external_data}
            assert list(i.dims)==desc['shape'] and i.data_type==onnx.TensorProto.FLOAT
            assert ext['location']==WEIGHTS.name and int(ext['offset'])==desc['offset'] and int(ext['length'])==desc['bytes']
    jobs=[]
    for request in source['requests']:
        cells={}
        for job in [j for j in source['schedule'] if j['request']==request['request']]:
            folder=PRIOR/'outputs'/job['id'];result=folder/'result.json';files[rel(result)]=pin(result)
            for record in read(result)['records']:cells[record['kind']]=(folder,record)
        assert set(cells)==set(CELLS)
        for projection,pj in PROJECTIONS.items():
            for cell in CELLS:
                folder,record=cells[cell];descriptors={}
                for role,index in [('input',pj['input']),('actual',pj['output'])]:
                    value=record['outputs'][index];path=folder/value['file'];identity=pin(path)
                    assert value['index']==index and value['name']==source['outputs'][index]['name']
                    assert identity==dict(bytes=math.prod(value['shape'])*4,sha256=value['sha256'])
                    assert value['shape']==[1,1500,1280 if role=='input' else pj['shape'][1]]
                    files[rel(path)]=identity;descriptors[role]=dict(file=rel(path),pin=identity,shape=value['shape'])
                jobs.append(dict(id=f"r{request['request']:02}-{projection}-{cell}",request=request['request'],selected_request=request['selected_request'],
                    name=request['name'],features=request['features'],projection=projection,cell=cell,**descriptors))
    assert len(jobs)==64 and len({j['id'] for j in jobs})==64
    for path in sorted(Path(__file__).parent.glob('*.py')):files[rel(path)]=pin(path)
    files[rel(Path(__file__).with_name('README.md'))]=pin(Path(__file__).with_name('README.md'))
    native=[p for p in (Path(np.__file__).resolve().parent.parent/'numpy.libs').glob('*.dll')]
    numeric={str(p):pin(p) for p in native+[Path(np.__file__).resolve(),Path(sys.executable)]}
    manifest=dict(schema=1,protocol='natural-whisper-fp64-projections-v1',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        prior_receipt=RECEIPT,prior_files_verified=len(old['files'])+len(old['reports']),files=files,numerical_files=numeric,numpy=np.__version__,
        limits=LIMITS,thread_environment=THREAD_ENV,projections=PROJECTIONS,jobs=jobs,reference_bytes=sum(math.prod(j['actual']['shape'])*8 for j in jobs))
    assert manifest['reference_bytes']==2457600000
    write(base/'manifest.json',manifest)
    print(json.dumps(dict(jobs=len(jobs),prior_files=manifest['prior_files_verified'],manifest=pin(base/'manifest.json'))))

if __name__=='__main__':main()
