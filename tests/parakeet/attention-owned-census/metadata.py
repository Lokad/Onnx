"""Read exact initializer bytes from the already downloaded, pinned export."""
import hashlib
import json
from pathlib import Path
import onnx

ROOT=Path(__file__).resolve().parents[3]


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def derive(manifest,old):
    directory=ROOT/'models/parakeet-tdt-0.6b-v3'
    for name in ['encoder-model.onnx','encoder-model.onnx.data']:
        expected={k:manifest['models'][name][k] for k in ['bytes','sha256']}
        assert pin(directory/name)==expected==old['external'][manifest['models'][name]['path']]
    model=onnx.load(str(directory/'encoder-model.onnx'),load_external_data=False)
    initializers={t.name:t for t in model.graph.initializer}
    projection=ROOT/'tests/parakeet/pointwise-tail-profile-results/projection-breakdown-20260927.json'
    routes=read(ROOT/'tests/parakeet/pointwise-tail-profile-results/attention-routes-20260927.json')
    assert routes['closure']['sha256']=='e83b34fb3aef0dacfc387c919638fd710fa266cd11a315fb6494195cd29ebc72'
    assert pin(projection)==routes['inputs'][projection.relative_to(ROOT).as_posix()]
    records=[r for r in read(projection)['records'] if r['group']!='Stem output projection']
    assert len(records)==216 and len({r['weight']['name'] for r in records})==216
    maps={r['Name'].removeprefix('packed:') for r in old['retained'] if r['PackedHash'] is not None}
    assert len(maps)==37
    result=[]
    with (directory/'encoder-model.onnx.data').open('rb') as stream:
        for row in records:
            name=row['weight']['name']; tensor=initializers[name]
            shape=list(tensor.dims)
            assert shape==row['weight']['dims'] and tensor.data_type==onnx.TensorProto.FLOAT
            assert not tensor.raw_data and not tensor.float_data
            external={r.key:r.value for r in tensor.external_data}
            assert set(external)=={'location','offset','length'} and external['location']=='encoder-model.onnx.data'
            offset,length=int(external['offset']),int(external['length'])
            assert length==shape[0]*shape[1]*4 and offset>=0
            stream.seek(offset);data=stream.read(length);assert len(data)==length
            result.append(dict(Name=name,Bytes=length,Hash=hashlib.sha256(data).hexdigest(),Shape=shape,Cached=name in maps))
    weights={r['Name']:r for r in result}
    assert all(weights[r['Name']]==r for r in old['weights']), 'All 96 prior logical hashes must match direct export bytes'
    attention=[r for r in result if r['Shape']==[1024,1024]]
    assert len(attention)==120 and sum(not r['Cached'] for r in attention)==92
    assert sum(not r['Cached'] for r in result)==179
    return dict(weights=result,onnx_version=onnx.__version__,model_pins={n:pin(directory/n)
        for n in ['encoder-model.onnx','encoder-model.onnx.data']},projection=pin(projection),
        source='float32 little-endian initializer external-data ranges; no inference or model download',
        original_feed_forward_hashes_exact=True,added_attention_weights=92)
