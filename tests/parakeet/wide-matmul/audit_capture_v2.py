"""Check real operand provenance, layouts and immutable frozen inputs."""
from common import *
import numpy as np
import onnx


def main():
    assert not (BASE/'capture-closed.json').exists()
    prepared=read(BASE/'prepared.json');assert prepared['passed'];verify(prepared['files'])
    state=read(BASE/'capture-state.json');assert state['complete'] and state['passed'] and state['code']==0
    for key in ('supervisor','worker'):terminal(state[key])
    resources=[json.loads(s) for s in (BASE/'capture-resources.jsonl').read_text().splitlines()]
    assert len(resources)==state['samples']>0 and max(v['rss'] for v in resources)==state['peak_rss']
    assert state['preflight']['available']>=14*1024**3
    assert all(v['seconds']<600 and v['rss']<12*1024**3 and v['available']>=1024**3 and v['disk']>=20*1024**3 and v['affinity']==[2] and v['bytes']<=512*1024**2 for v in resources)
    result=read(BASE/'capture/result.json');assert result['passed']
    identity=result['identity'];assert identity['runtime']=='.NET 10.0.12' and identity['affinity']==4 and identity['processor_count']==1 and identity['flags']=={}
    assert identity['core_sha256']==pin(BASE/'bin/Lokad.Onnx.dll')['sha256'] and identity['runner_sha256']==pin(BASE/'bin/Probe.dll')['sha256']
    spec=read(BASE/'capture-manifest.json');cases=spec['cases'];assert [c['frames'] for c in cases]==[51,106,167,225]
    assert result['requests']==[dict(name=c['name'],frames=c['frames'],complete_encoder_matches=True,inputs_unchanged=True) for c in cases]
    graphs=read(TRACE/'trace-output/graphs.json');nodes={n['id']:n for n in graphs['encoder']}
    assert [(e['name'],e['node']) for e in result['entries']]==[(c['name'],n) for c in cases for n in [99,102,109]]
    arrays={}
    def tensor(value):
        p=(BASE/'capture'/value['file']).resolve();assert p.parent==(BASE/'capture').resolve() and p.suffix=='.bin'
        assert value['dtype']=='<f4' and pin(p)=={k:value[k] for k in ['bytes','sha256']}
        v=np.fromfile(p,dtype='<f4');assert v.size==int(np.prod(value['shape'])) and np.isfinite(v).all()
        arrays[value['file']]=dict(path=p.relative_to(ROOT).as_posix(),**pin(p))
        return v.reshape(value['shape'])
    model_path=ROOT/spec['encoder']['path'];model=onnx.load(model_path,load_external_data=False)
    initializers={i.name:i for i in model.graph.initializer}
    weights={}
    for name,value in result['weights'].items():
        a=tensor(value); proto=initializers[name]
        assert proto.data_type==onnx.TensorProto.FLOAT and proto.data_location==onnx.TensorProto.EXTERNAL
        external={v.key:v.value for v in proto.external_data}
        source=(model_path.parent/external['location']).resolve()
        allowed={str((ROOT/v['path']).resolve()) for v in spec['models']}
        assert str(source) in allowed and source.is_relative_to(model_path.parent.resolve())
        offset=int(external.get('offset','0')); length=int(external['length'])
        assert offset>=0 and length==int(np.prod(proto.dims))*4 and offset+length<=source.stat().st_size
        # Read only the selected range of the already digest-verified local
        # asset. ONNX 1.22's general loader refuses our retained hard links.
        with source.open('rb') as stream: stream.seek(offset); raw=stream.read(length)
        assert len(raw)==length
        b=np.frombuffer(raw,dtype='<f4').reshape(tuple(proto.dims))
        assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),name
        weights[name]=a
    for e in result['entries']:
        node=nodes[e['node']];assert e['node_name']==node['name'] and e['weight']==node['inputs'][1]
        a=tensor(e['a']);y=tensor(e['y']);b=weights[e['weight']]
        assert a.shape==(1,e['m'],e['k']) and b.shape==(e['k'],e['n']) and y.shape==(1,e['m'],e['n'])
        for value in [e['a'],e['y']]:
            s=value['shape'];assert not value['reverse'] and value['strides']==[s[1]*s[2],s[2],1]
    assert len(arrays)==27 and len(result['weights'])==3
    assert {p.name for p in (BASE/'capture').iterdir() if p.is_file()}==set(arrays)|{'result.json'}
    save(BASE/'probe-manifest.json',dict(capture=dict(path=(BASE/'capture/result.json').relative_to(ROOT).as_posix(),**pin(BASE/'capture/result.json')),arrays=arrays))
    failed=ROOT/'artifacts/parakeet-wide-matmul-20260921'
    assert pin(failed/'failure.json')['sha256']=='b4a979cc58bd4f13e2e4a34cc1417745030abfdaa4f15b73d16f7eb6a25a931f'
    for name,wanted in read(failed/'failure.json')['files'].items():assert pin(failed/name)==wanted
    files=dict(prepared['files'])
    for p in [BASE/'prepared.json',BASE/'capture-state.json',BASE/'capture-resources.jsonl',BASE/'capture-stdout.txt',BASE/'capture-stderr.txt',BASE/'probe-manifest.json',failed/'failure.json',TOOLS/'audit_capture_v2.py',BASE/'capture-audit-failure.json']:
        files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in (BASE/'capture').iterdir():
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    save(BASE/'capture-closed.json',dict(passed=True,files=files,identities=[state[k] for k in ('supervisor','worker')],independent_onnx_weights=True))
    print(json.dumps(dict(passed=True,requests=4,fixtures=12,arrays=27,layouts='standard row-major',closed=pin(BASE/'capture-closed.json'))))


if __name__=='__main__':main()
