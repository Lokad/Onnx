"""Launch and validate the shared complete-output graph benchmark callers."""
from array import array
import datetime as dt
import math
from pathlib import Path
import sys,time

import benchmark_worker as common
import campaign_evidence as evidence

RUNNER='Lokad.Onnx.GraphBench.dll'
CASES=('e5-30tok','dinov3','resnet50','gpt2')


def floats(path):
    values=array('f');values.frombytes(path.read_bytes())
    if sys.byteorder!='little':values.byteswap()
    return values


def validate(directory, build, name, engine, mode, manifest, native_versions):
    import json
    value=json.loads((directory/'result.json').read_text(encoding='utf-8-sig'))
    evidence.require(value['key']==name and value['mode']==mode and value['passed']
                     and value['inputs_unchanged'] and value['held_outputs_unchanged'], 'graph numerical/ownership check failed')
    if engine=='lok':
        evidence.require(value['core']==build['core_sha256'], 'graph core changed')
        evidence.require(value['consumer']==evidence.sha256(build['runner']), 'graph caller changed')
        evidence.require(value['runtime'].startswith(build['target_framework'][3:]+'.'), 'graph runtime changed')
        evidence.require(value['flags']=={}, 'graph runtime override')
    else:
        evidence.require(value['native_versions']==native_versions, 'native graph versions changed')
    case=next(c for c in manifest['cases'] if c['key']==name)
    evidence.require([a['name'] for a in value['arrays']]==[a['name'] for a in case['outputs']], 'graph output names changed')
    for actual,expected in zip(value['arrays'],case['outputs'],strict=True):
        filename=actual['file']
        evidence.require(Path(filename).name==filename and '\\' not in filename, 'invalid graph array filename')
        evidence.require(actual['shape']==expected['shape'] and actual['values']==math.prod(expected['shape']), 'graph output shape changed')
        evidence.require((directory/filename).stat().st_size==4*actual['values'] and evidence.sha256(directory/filename)==actual['sha256'], 'graph array changed')
    clocks=value['clocks'];warmups=1200 if name=='e5-30tok' else 600
    expected_count=3 if mode=='verify' else warmups+180
    evidence.require(value['calls']==len(clocks)==expected_count, 'graph call count changed')
    for i,clock in enumerate(clocks):
        evidence.require(clock['index']==i and clock['warmup']==(mode=='verify' or i<warmups), 'graph call order changed')
        evidence.require(type(clock['ticks']) is int and clock['ticks']>0 and type(clock['frequency']) is int and clock['frequency']>0, 'invalid graph clock')
    return value,[clock['ticks']*1000/clock['frequency'] for clock in clocks if not clock['warmup']]


def compare(managed, native):
    import json
    left=json.loads((managed/'result.json').read_text(encoding='utf-8-sig'))
    right=json.loads((native/'result.json').read_text(encoding='utf-8-sig'))
    evidence.require([a['name'] for a in left['arrays']]==[a['name'] for a in right['arrays']], 'native graph names differ')
    count=0;worst=0.
    for a,b in zip(left['arrays'],right['arrays'],strict=True):
        evidence.require(a['shape']==b['shape'] and a['values']==b['values'], 'native graph shape differs')
        evidence.require(evidence.sha256(managed/a['file'])==a['sha256'] and evidence.sha256(native/b['file'])==b['sha256'], 'complete graph array changed')
        x,y=floats(managed/a['file']),floats(native/b['file'])
        evidence.require(len(x)==len(y)==a['values'] and all(math.isfinite(v) for v in x) and all(math.isfinite(v) for v in y), 'native graph extent/finite check failed')
        worst=max(worst,max((abs(float(u)-v)/max(1,abs(float(v))) for u,v in zip(x,y,strict=True)),default=0.))
        count+=len(x)
    evidence.require(worst<=1e-4, 'complete graph/native arrays exceed1e-4')
    return dict(values=count,maximum_scaled_error=worst)


def child(build, output, name, engine, mode, pair, manifest_path, manifest, native_versions, env, cpu, dotnet, python, index, deadline):
    evidence.require(time.monotonic()<deadline, 'full benchmark duration limit exceeded before launch')
    stem=f'graph-{index:03d}-{name}-{engine}-{mode}'
    destination=output/stem
    command=([str(dotnet),str(build['runner'])] if engine=='lok' else [str(python),str(Path(__file__).with_name('graph_native.py'))])
    command += [str(manifest_path),name,str(destination),mode]
    started=dt.datetime.now(dt.timezone.utc);failure=None
    with (output/(stem+'.log')).open('x',encoding='utf-8') as log:
        timeout=min(600,deadline-time.monotonic())
        evidence.require(timeout>0, 'full benchmark duration limit exceeded before launch')
        process=common.spawn_child(command,cpu,cwd=output,env=env,stdout=log,stderr=log)
        try:code=common.wait_child(process,stem,timeout=timeout)
        except BaseException as error:failure=error;code=process.returncode
    common.save_json(output/(stem+'.supervision.json'),dict(command=command,pid=process.pid,exit_code=code,
        started_utc=started.isoformat(),completed_utc=dt.datetime.now(dt.timezone.utc).isoformat(),failure=None if failure is None else str(failure)))
    if failure is not None:raise failure
    evidence.require(code==0,'graph worker failed: '+stem)
    value,samples=validate(destination,build,name,engine,mode,manifest,native_versions)
    return dict(case=name,engine=engine,pair=pair,phase='conformance' if mode=='verify' else 'timing',aggregation='mean',samples_ms=samples,
        result=stem+'/result.json',result_sha256=evidence.sha256(destination/'result.json'),supervision=stem+'.supervision.json',
        native_signature=value.get('native_module'),setup_seconds=value['setup_seconds'])
