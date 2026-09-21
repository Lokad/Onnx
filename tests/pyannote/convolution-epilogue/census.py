"""Audit actual tiled paths and bias work from frozen metadata; no inference."""
import hashlib, json, math, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-convolution-epilogue-20260921'
MODEL=ROOT/'artifacts/pyannote-convolution-portable-rows-20260921'
CENSUS=ROOT/'artifacts/pyannote-convolution-allocation-20260921'
NATIVE=ROOT/'artifacts/pyannote-epilogue-source-20260921'
sys.path.insert(0,str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil

def read(p):return json.loads(p.read_text(encoding='utf8'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def rel(p):return p.relative_to(ROOT).as_posix()
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')

def main():
    assert not BASE.exists()
    owner=psutil.Process();old_affinity=owner.cpu_affinity();owner.cpu_affinity([0])
    try:
        import onnx
        files={};trusted={}
        for p,sha in [(MODEL/'closed.json','27ff741eb0a6aceb354806a030730175a95e429d24ae5c307a31a7633b756669'),
                      (CENSUS/'closed.json','ff0eee55371edfee4704979dda915fe1d69bcb0de8e708d32be4f720e69c8bc4')]:
            assert pin(p)['sha256']==sha
            closure=read(p);assert closure['passed']
            trusted.update({str((ROOT/k).resolve()):v for k,v in closure['files'].items()})
            files[rel(p)]=pin(p)
        def checked(p):
            actual=pin(p);assert actual==trusted[str(p.resolve())],p;files[rel(p)]=actual;return p
        manifest=read(checked(MODEL/'manifest.json'))
        model_path=checked(ROOT/manifest['models']['embedding']['path'])
        model=onnx.load(model_path,load_external_data=False)
        initializers={t.name:t for t in model.graph.initializer}
        conv={n.name:n for n in model.graph.node if n.op_type=='Conv'}
        old=read(checked(CENSUS/'analysis.json'))
        source_path=checked(MODEL/'source/src/Lokad.Onnx/TensorOps.ConvPool.cs')
        source=source_path.read_text(encoding='utf8')
        for literal in ['kH == 1 && kW == 1 && sH == 1 && sW == 1 && dH == 1 && dW == 1',
                        'outH == H && outW == W',
                        'float v = hasBias ? ds[blkRow + j] + bi : ds[blkRow + j];']:
            assert source.count(literal)==1,literal
        native_receipt=read(NATIVE/'source.json')
        assert native_receipt['status']==200 and native_receipt['revision']=='2e2543fbe9fae542f921d47a72d21d5a4ef0b710'
        assert pin(NATIVE/'activate.cpp')=={k:native_receipt[k] for k in ['bytes','sha256']}
        for p in [NATIVE/'source.json',NATIVE/'activate.cpp']:files[rel(p)]=pin(p)
        prior_shapes=ROOT/'artifacts/pyannote-portable-row-groups-v3-20260921/shapes.json'
        prior_closed=read(ROOT/'artifacts/pyannote-portable-row-groups-v3-20260921/closed.json')
        assert pin(prior_shapes)==prior_closed['files'][rel(prior_shapes)]
        files[rel(prior_shapes)]=pin(prior_shapes)
        old_shapes={tuple(r[k] for k in ['m','n','k']) for r in read(prior_shapes)['shapes']}
        cases=[];all_shapes=set()
        for case in old['cases']:
            rows=[]
            assert {r['name'] for r in case['rows']}==set(conv)
            for row in case['rows']:
                node=conv[row['name']];attrs=row['attributes']
                assert attrs['group']==1
                m,c,kh,kw=row['weight_shape'];batch,channels,h,w=row['input_shape'];_,_,oh,ow=row['output_shape']
                assert channels==c and batch==1
                assert len(node.input)==3 and list(initializers[node.input[2]].dims)==[m]
                assert initializers[node.input[2]].data_type==onnx.TensorProto.FLOAT
                pointwise=(kh==kw==1 and attrs['strides']==[1,1] and attrs['dilations']==[1,1]
                    and attrs['pads']==[0,0,0,0] and (h,w)==(oh,ow))
                reduction=c*kh*kw;columns=oh*ow
                block=columns if reduction*columns<=65536 else min(columns,max(32,(65536//(reduction+m))//32*32))
                tiled=not pointwise and block<columns
                widths=([block]+([columns%block] if columns%block else [])) if tiled else []
                for k in widths:all_shapes.add((m,reduction,k))
                rows.append(dict(name=row['name'],input_shape=row['input_shape'],weight_shape=row['weight_shape'],
                    output_shape=row['output_shape'],attributes=attrs,bias=node.input[2],bias_shape=[m],
                    pointwise=pointwise,tiled=tiled,block_columns=block,tile_widths=widths,
                    output_values=batch*m*columns,bias_additions=batch*m*columns,
                    tiled_copy_values=batch*m*columns if tiled else 0))
            cases.append(dict(name=case['name'],rows=rows,tiled_nodes=sum(r['tiled'] for r in rows),
                biased_nodes=len(rows),bias_additions=sum(r['bias_additions'] for r in rows),
                tiled_copy_values=sum(r['tiled_copy_values'] for r in rows)))
        assert len(all_shapes)==22 and len(old_shapes)==16 and old_shapes<all_shapes
        assert all(c['tiled_nodes']==c['biased_nodes']==36 and c['bias_additions']==39941120 for c in cases)
        missing=sorted(all_shapes-old_shapes)
        assert missing==[(64,32,472),(64,32,672),(128,64,200),(128,64,320),(256,128,130),(256,128,160)]
        BASE.mkdir()
        analysis=dict(passed=True,inference_executed=False,onnx=onnx.__version__,cases=cases,
            all_tile_shapes=[dict(m=m,n=n,k=k) for m,n,k in sorted(all_shapes)],missing_from_original_kernel_probe=[dict(m=m,n=n,k=k) for m,n,k in missing],
            scope='Static workload coverage and bias/copy census; no JIT instruction or latency attribution.',
            supervisor=dict(pid=owner.pid,birth=owner.create_time()),affinity=owner.cpu_affinity(),
            resident_bytes_at_end=owner.memory_info().rss,available_bytes_at_end=psutil.virtual_memory().available)
        save(BASE/'analysis.json',analysis)
        files[rel(Path(__file__))]=pin(Path(__file__));files[rel(BASE/'analysis.json')]=pin(BASE/'analysis.json')
        save(BASE/'closed.json',dict(passed=True,files=files,analysis=pin(BASE/'analysis.json')))
        print(dict(tiled=36,bias_additions=39941120,shapes=22,missing=missing,closure=pin(BASE/'closed.json')))
    finally:owner.cpu_affinity(old_affinity)

if __name__=='__main__':main()
