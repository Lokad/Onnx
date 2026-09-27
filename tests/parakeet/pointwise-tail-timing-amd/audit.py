"""All clocks, original repeatability limits, and the fixed remainder prediction."""
import importlib.util
import json
import math
from pathlib import Path
import sys
from run import BASE,ROOT,TOOLS,pin,read,write,prepared

loader=importlib.util.spec_from_file_location('arithmetic_audit',TOOLS.parent/'pointwise-tail-arithmetic-contracts-amd/audit.py')
parent=importlib.util.module_from_spec(loader);loader.loader.exec_module(parent)


def analyze(times,spec):
    shapes=spec['shapes'];count=len(shapes);passes=spec['measurements'];controls=[];gates=[];means={}
    def control(name,values):
        assert values and min(values)>0 and all(math.isfinite(v) for v in values)
        ratio=max(values)/min(values);controls.append(dict(name=name,ratio=ratio,limit=spec['repeatability_limit'],passed=ratio<=spec['repeatability_limit']))
    for job in spec['jobs']:
        name=job['name'];rows=times[name];assert len(rows)==count and all(len(r)==passes for r in rows)
        means[name]=[math.fsum(r)/passes for r in rows]
        for i,row in enumerate(rows):control(name+'/shape/'+str(i),row)
        control(name+'/corpus',[math.fsum(rows[i][p] for i in range(count)) for p in range(passes)])
    product={}
    for role in ['baseline','candidate']:
        names=[j['name'] for j in spec['jobs'] if j['role']==role];assert len(names)==2
        for i in range(count):control(role+'/between/shape/'+str(i),[means[n][i] for n in names])
        control(role+'/between/corpus',[math.fsum(means[n]) for n in names])
        product[role]=[math.fsum(means[n][i] for n in names)/2 for i in range(count)]
    shape_rows=[]
    for i,shape in enumerate(shapes):
        before=product['baseline'][i];after=product['candidate'][i];ratio=after/before
        row=dict(shape,baseline_seconds=before,candidate_seconds=after,ratio=ratio,saved_seconds=before-after)
        shape_rows.append(row);gates.append(dict(name='shape/'+str(i),ratio=ratio,limit=spec['regression_limit'],passed=ratio<=spec['regression_limit']))
    actual=[i for i,s in enumerate(shapes) if not s['control']]
    before=math.fsum(product['baseline'][i] for i in actual);after=math.fsum(product['candidate'][i] for i in actual)
    gates.append(dict(name='actual-shapes-improve',ratio=after/before,limit=1.0,strict=True,passed=after<before))
    contrasts=[]
    for m in sorted({s['m'] for s in shapes}):
        indices={s['k']:i for i,s in enumerate(shapes) if s['m']==m}
        old=product['baseline'][indices[222]]-product['baseline'][indices[225]]
        new=product['candidate'][indices[222]]-product['candidate'][indices[225]]
        row=dict(m=m,baseline_222_minus_225_seconds=old,candidate_222_minus_225_seconds=new,passed=old>0 and new<old)
        contrasts.append(row);gates.append(dict(name='remainder-contrast/'+str(m),**row))
    return dict(component_admitted=all(r['passed'] for r in controls+gates),controls=controls,gates=gates,shapes=shape_rows,contrasts=contrasts,
        actual_shape_sum=dict(baseline_seconds=before,candidate_seconds=after,ratio=after/before),process_means=means)


def capture():
    folder,spec,state,built,resources=parent.collected('capture')
    assert read(BASE/'build-review.json')['passed']
    accounting_loader=importlib.util.spec_from_file_location('timing_accounting',BASE/'bundle/campaign_processes.py')
    accounting=importlib.util.module_from_spec(accounting_loader);accounting_loader.loader.exec_module(accounting)
    times={};output_hashes=None;clocks=0
    for declared,job in zip(spec['jobs'],state['runs'],strict=True):
        result=read(folder/'probe'/declared['name']/'result.json')
        assert result['completed'] and result['passed'] and result['role']==declared['role'] and result['inputs_immutable']
        assert result['pid']==job['owner']['pid'] and result['runtime']=='.NET 10.0.8' and not result['flags']
        assert result['core_sha256']==spec['products'][declared['role']]['Lokad.Onnx.dll']['sha256']
        assert result['consumer_sha256']==built['runtime'][declared['role']+'/TailContracts.dll']['sha256']
        checked=accounting.foreign_fraction(job['cpu_before'],job['cpu_after'],state['supervisor']['pid'])
        assert checked==job['accounting'] and checked['valid'] and checked['foreign_cpu_fraction']<=spec['foreign_cpu_limit']
        expected=[(phase,p,i,s) for phase,passes in [('warmup',spec['warmups']),('measured',spec['measurements'])] for p in range(passes) for i,s in enumerate(spec['shapes'])]
        assert len(result['records'])==len(expected) and type(result['frequency']) is int and result['frequency']>0
        measured=[[] for _ in spec['shapes']];hashes=[None for _ in spec['shapes']];total=0
        for (phase,p,i,shape),row in zip(expected,result['records'],strict=True):
            assert row['phase']==phase and row['pass']==p and row['index']==i and {k:row[k] for k in shape}==shape
            assert type(row['ticks']) is int and row['ticks']>0 and row['allocated_bytes']==0 and row['bit_exact']
            assert len(row['output_sha256'])==64
            if hashes[i] is None:hashes[i]=row['output_sha256']
            else:assert hashes[i]==row['output_sha256']
            if phase=='measured':measured[i].append(row['ticks']/result['frequency'])
            total+=row['ticks'];clocks+=1
        assert result['ended_ticks']-result['started_ticks']>=total
        if output_hashes is None:output_hashes=hashes
        else:assert output_hashes==hashes
        times[declared['name']]=measured
    assert clocks==1600
    performance=analyze(times,spec)
    value=dict(passed=True,performance=performance,clocks=clocks,measured_clocks=800,warmup_clocks=800,resources=resources,
        identities=spec['products'],boundary=spec['timing_boundary'],numerical_closure=spec['numerical_closure'],codegen_review=spec['codegen_review'],
        all_outputs_exact=True,release_admitted=False,reviewer=pin(Path(__file__)))
    write(BASE/'analysis.json',value)
    write(BASE/'closed.json',dict(completed=True,component_admitted=performance['component_admitted'],analysis=pin(BASE/'analysis.json'),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),component_admitted=performance['component_admitted'],clocks=clocks,
        controls=[sum(r['passed'] for r in performance['controls']),len(performance['controls'])],gates=[sum(r['passed'] for r in performance['gates']),len(performance['gates'])],
        actual_shape_sum=performance['actual_shape_sum'],contrasts=performance['contrasts'])))


if __name__=='__main__':
    if sys.argv[1]=='build':parent.build()
    elif sys.argv[1]=='capture':capture()
    else:raise ValueError(sys.argv[1])
