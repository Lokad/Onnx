"""Freeze the selected traces against complete closed inputs and actual runtime bytes."""
from pathlib import Path
import argparse,copy,hashlib,json,shutil,subprocess
import numpy as np,onnx
from common import ROOT,CORE,ORIGIN,TRACE,PROTOCOL,pin,read,write,schedule

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args();base=args.artifact.resolve()
    assert not (base/'manifest.json').exists() and shutil.disk_usage(base).free>=25*1024**3
    old=ROOT/'artifacts/whisper-input-cross-isolated-20260920';assert pin(old/'closed.json')['sha256']==ORIGIN
    closed=read(old/'closed.json');previous=read(old/'manifest.json');assert pin(old/'manifest.json')==closed['files']['manifest.json']
    frames=ROOT/'artifacts/whisper-frame-distribution-20260920';assert pin(frames/'closed.json')['sha256']=='51fe593de2da8fcab96e44faa5dd61151ed3da4d8ee35a060da2985a5572eb7f'
    files={}
    def bind(path,expected=None):
        value=pin(path)
        if expected is not None:assert value==expected,path
        files[path.relative_to(ROOT).as_posix()]=value
    for path in [old/'closed.json',old/'manifest.json',frames/'closed.json',ROOT/'tests/Shared/NpySupport.cs']:bind(path)
    for name in ['audit.json', 'campaign.json']:bind(old/name,closed['files'][name])
    original=ROOT/previous['model'];trace=original.with_name('encoder_trace_20260918.onnx')
    bind(original,previous['files'][previous['model']]);bind(Path(str(original)+'_data'),previous['files'][str(Path(str(original)+'_data').relative_to(ROOT)).replace('\\','/')])
    bind(trace);assert pin(trace)['sha256']==TRACE
    model=onnx.load(original,load_external_data=False);traced=onnx.load(trace,load_external_data=False)
    changed=onnx.ModelProto();changed.CopyFrom(model);del changed.graph.output[:];changed.graph.output.extend(traced.graph.output)
    assert changed.SerializeToString(deterministic=True)==traced.SerializeToString(deterministic=True) and len(traced.graph.node)==1559
    outputs=[]
    for value in traced.graph.output:
        assert value.type.tensor_type.elem_type==onnx.TensorProto.FLOAT
        shape=[d.dim_value if d.HasField('dim_value') else 1 for d in value.type.tensor_type.shape.dim]
        assert all(d.HasField('dim_value') or d.dim_param=='batch_size' for d in value.type.tensor_type.shape.dim)
        outputs.append(dict(name=value.name,shape=shape))
    assert len(outputs)==41 and outputs[-1]['name']=='last_hidden_state'
    assert [o['shape'] for o in outputs]==[[1,1280,3000]]*2+[[1,1280,1500]]*2+[[1,1500,1280]]*37
    requests=[]
    for index,original_index in enumerate([0,10,9,20]):
        item=copy.deepcopy(previous['requests'][original_index]);item['request']=index;item['original_request']=original_index;item['baselines']={}
        for key in ['managed_features','native_features']:
            name=item[key]['file'];bind(ROOT/name,previous['files'][name])
        for engine in ['managed','native']:
            folder=old/'outputs'/f"{engine}-{original_index:02}-{item['name']}";result=read(folder/'result.json')
            bind(folder/'result.json',closed['files'][(folder/'result.json').relative_to(old).as_posix()])
            for record in result['records']:
                path=folder/record['file'];bind(path,closed['files'][path.relative_to(old).as_posix()])
                assert record['shape']==[1,1500,1280] and pin(path)['sha256']==record['sha256']
                item['baselines'][record['kind']]=dict(file=path.relative_to(ROOT).as_posix(),format='f32',shape=record['shape'],raw_sha256=record['sha256'])
        assert set(item['baselines'])=={'MM','MN','NM','NN'}
        # Unused older hidden references are removed; every remaining referenced file is bound above.
        del item['managed_hidden'];del item['native_hidden'];requests.append(item)
    assert pin(base/'bin/Lokad.Onnx.dll')['sha256']==CORE
    assert '0 Warning(s)' in (base/'build.log').read_text() and read(base/'serialization-check.json')['passed'] is True
    assert 'OK' in (base/'unit-tests.log').read_text()
    for name in ['build.log','unit-tests.log','serialization-check.json']:bind(base/name)
    for path in (base/'bin').iterdir():
        if path.is_file():bind(path)
    for path in Path(__file__).parent.iterdir():
        if path.is_file():bind(path)
    for path in [ROOT/'tests/whisper/input-cross/audit.py',ROOT/'tests/whisper/input-cross/common.py']:bind(path)
    for name,want in previous['native_runtime']['files'].items():assert pin(Path(name))==want
    spec=dict(schema=1,protocol=PROTOCOL,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        model=trace.relative_to(ROOT).as_posix(),original_model=previous['model'],requests=requests,outputs=outputs,core_sha256=CORE,
        native_runtime=previous['native_runtime'],native_serialization=dict(file='optimized.onnx',external_file='optimized.weights',minimum_external_bytes=1024),
        files=files,limits=dict(seconds=1800,rss=8*1024**3,available=1024**3,preflight_available=10*1024**3),preflight_wait_seconds=900,
        workers=8,calls=16,arrays=656,output_payload_bytes=5283840000,scaled_error_limit=1e-4)
    spec['schedule']=schedule(spec);write(base/'manifest.json',spec)
    print('Frozen',len(files),'files;',pin(base/'manifest.json'))

if __name__=='__main__':main()
