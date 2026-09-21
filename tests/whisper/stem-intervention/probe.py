"""Prepare rounded reference states, then run the unchanged encoder suffix."""
import argparse
import collections
import json
import shutil
import subprocess
import time
from protocol import *
import onnx


def prepare():
    assert not BASE.exists() and shutil.disk_usage(ROOT).free >= LIMITS['disk']
    closure = PRIOR/'closed.json'
    assert pin(closure)['sha256'] == '6baa0d3774515baf452a538e553b556a83d6c486f10b43895ab02b775a2baee9'
    closed = read(closure); assert closed['structural_passed']
    old = read(PRIOR/'manifest.json'); files = dict(old['files'])
    for name, expected in old['files'].items():
        assert pin(ROOT/name) == expected, name
    for name, expected in closed['files'].items():
        path = PRIOR/name; assert pin(path) == expected, name
        files[rel(path)] = expected
    for name, expected in closed['reports'].items():
        assert pin(ROOT/name) == expected, name
        files[name] = expected
    for name, expected in old['numerical_files'].items():
        assert pin(name) == expected, name
    assert all(absent(b) for b in closed['births'])
    files[rel(closure)] = pin(closure)
    model = onnx.load(ROOT/old['model'],load_external_data=False)
    start,nodes = select_suffix(model,CUT)
    assert start == 14 and len(nodes) == 1545 and old['outputs'][4]['name'] == CUT
    assert [r['original'] for r in old['requests']] == SELECTED
    BASE.mkdir(); (BASE/'inputs').mkdir()
    requests = []
    for request in old['requests']:
        arrays = {}
        for engine in ['numpy','ort']:
            desc = request['references'][engine][4]
            arrays[engine] = np.fromfile(ROOT/desc['file'],dtype='<f8').reshape(desc['shape'])
        incoming = narrow_stem(arrays['numpy']); other = narrow_stem(arrays['ort'])
        input_path = BASE/'inputs'/f"{request['index']:02}.f32"
        with input_path.open('xb') as stream:
            incoming.tofile(stream)
        files[rel(input_path)] = pin(input_path)
        detail = dict(numpy_reference=request['references']['numpy'][4],ort_reference=request['references']['ort'][4],
                      cast_error=metric(incoming,arrays['numpy']),
                      rounded_reference_differences=int(np.count_nonzero(incoming.view(np.uint32)!=other.view(np.uint32))),
                      rounded_reference_max_absolute=float(np.abs(incoming-other).max()))
        assert detail['cast_error']['max_scaled'] <= 1e-7
        requests.append(dict(index=request['index'],original=request['original'],name=request['name'],features=request['input'],
            input=dict(file=rel(input_path),format='f32',shape=list(incoming.shape),raw_sha256=raw(incoming)),stem=detail,
            references={e:request['references'][e][5:] for e in ['numpy','ort']},baselines=request['baselines']))
    assert raw(source(requests[0]['input'])) == raw(source(requests[-1]['input']))
    for path in list(Path(__file__).parent.glob('*.py'))+[PRIOR_TOOLS/'protocol.py',PRIOR_TOOLS/'run.py',TOOLS/'common.py',TOOLS/'interpreter.py',TOOLS/'worker.py']:
        files[rel(path)] = pin(path)
    write(BASE/'manifest.json',dict(protocol=PROTOCOL,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        files=files,numerical_files=old['numerical_files'],model=old['model'],cut=CUT,node_offset=start,nodes=len(nodes),
        census=dict(collections.Counter(n.op_type for n in nodes)),outputs=old['outputs'][5:],output_offset=5,
        requests=requests,jobs=old['jobs'],limits=LIMITS,threads=THREADS,reference_limit=1e-4,scalar_limit=1e-7,
        useful_maximum_ratio=.5,controls=rel(PRIOR/'analysis.json')))
    print(json.dumps(dict(prepared=True,jobs=8,outputs=36,files=len(files),manifest=pin(BASE/'manifest.json'))))


def worker(job_id):
    spec=read(BASE/'manifest.json');assert spec['protocol']==PROTOCOL and spec['cut']==CUT
    job=next(j for j in spec['jobs'] if j['id']==job_id);request=spec['requests'][job['request']]
    folder=BASE/'outputs'/job_id;folder.mkdir(parents=True,exist_ok=False)
    state=source(request['input']);before=raw(state);saved={};started=time.monotonic()
    wanted={o['name']:(i+spec['output_offset'],o) for i,o in enumerate(spec['outputs'])}
    def capture(name,value):
        if name not in wanted:return
        index,desc=wanted[name];assert name not in saved and list(value.shape)==desc['shape'] and value.dtype==np.float32
        path=folder/f'{index:02}.f32'
        with path.open('xb') as stream:np.ascontiguousarray(value).tofile(stream)
        saved[name]=dict(index=index,name=name,shape=desc['shape'],file=path.name,pin=pin(path))
        print(json.dumps(dict(boundary=index,seconds=time.monotonic()-started)),flush=True)
    model=onnx.load(ROOT/spec['model'],load_external_data=False)
    records,dots=run_suffix(model,CUT,state,job['mode'],capture)
    assert len(records)==spec['nodes'] and set(saved)==set(wanted) and raw(state)==before
    runtime_spec=importlib.util.spec_from_file_location('qualified_reference_runtime',TOOLS/'worker.py')
    module=importlib.util.module_from_spec(runtime_spec);runtime_spec.loader.exec_module(module)
    write(folder/'result.json',dict(complete=True,job=job,manifest=pin(BASE/'manifest.json'),input_unchanged=True,input_sha256=before,
        runtime=module.runtime('numpy'),records=records,scalar_checks=dots,outputs=[saved[o['name']] for o in spec['outputs']],
        seconds=time.monotonic()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['prepare','worker']);parser.add_argument('--job')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    else:worker(args.job)
