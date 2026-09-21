"""Audit the complete intervention and compare with the unchanged prior controls."""
import collections
import datetime
import json
import math
from protocol import *


def checked_metric(a,b):
    whole,chunk=metric(a,b),metric(a,b,True)
    for key in ['max_scaled','failed_values','values']:assert whole[key]==chunk[key]
    assert math.isclose(whole['squared_error'],chunk['squared_error'],rel_tol=1e-12,abs_tol=1e-18)
    return whole


def main():
    assert not (BASE/'closed.json').exists() and not (BASE/'analysis.json').exists()
    spec=read(BASE/'manifest.json');state=read(BASE/'processes.json')
    assert spec['protocol']==PROTOCOL and spec['cut']==CUT and spec['node_offset']==14 and spec['nodes']==1545
    assert spec['limits']==LIMITS and len(spec['outputs'])==36 and spec['output_offset']==5
    assert state['complete'] and state['code']==0 and state['manifest']==pin(BASE/'manifest.json')
    assert len(state['runs'])==len(spec['jobs'])==8
    assert [r['original'] for r in spec['requests']]==SELECTED
    births=[state['supervisor']]+[r['worker'] for r in state['runs']];assert all(absent(b) for b in births)
    for name,want in spec['files'].items():assert pin(ROOT/name)==want,name
    for name,want in spec['numerical_files'].items():assert pin(name)==want,name
    controls=read(ROOT/spec['controls']);assert controls['structural_passed']
    rows=[];pins={};samples_count=0;peak=0;scalar_count=0
    for job,run in zip(spec['jobs'],state['runs'],strict=True):
        assert run['job']==job and run['complete'] and run['code']==0 and run['samples']>0
        request=spec['requests'][job['request']];incoming=source(request['input'])
        reference=request['stem']['numpy_reference'];ideal=np.fromfile(ROOT/reference['file'],dtype='<f8').reshape(reference['shape'])
        assert raw(incoming)==raw(narrow_stem(ideal))
        assert checked_metric(incoming,ideal)==request['stem']['cast_error']
        folder=BASE/'outputs'/job['id'];result=read(folder/'result.json')
        assert result['complete'] and result['job']==job and result['manifest']==pin(BASE/'manifest.json')
        assert result['input_unchanged'] and result['input_sha256']==request['input']['raw_sha256']
        runtime=result['runtime'];assert {k:runtime[k] for k in ['pid','birth']}==run['worker']
        assert runtime['affinity']==[2] and runtime['blas_threads']==1 and runtime['native_loaded'] is False
        for name,want in runtime['loaded'].items():assert want==spec['numerical_files'][name],name
        assert len(result['records'])==1545 and [r['index'] for r in result['records']]==list(range(14,1559))
        assert dict(collections.Counter(r['op'] for r in result['records']))==spec['census']
        assert all(r['dtype']=='float32' for r in result['records'])
        count=spec['census']['MatMul'] if job['mode']=='wide-matmul' else 0
        assert len(result['scalar_checks'])==count
        for item in result['scalar_checks']:
            assert len(item['checks'])==3
            for check in item['checks']:
                error=abs(check['actual']-check['expected'])/max(1.,abs(check['expected']))
                assert error==check['max_scaled'] and error<=spec['scalar_limit'];scalar_count+=1
        samples=[json.loads(line) for line in (BASE/'process'/job['id']/'samples.jsonl').read_text().splitlines()]
        assert len(samples)==run['samples'] and max(s['rss'] for s in samples)==run['peak_rss']
        for sample in samples:
            assert {k:sample[k] for k in ['pid','birth']}==run['worker'] and sample['affinity']==[2]
            assert 0<=sample['seconds']<LIMITS['seconds'] and sample['rss']<LIMITS['rss']
            assert sample['available']>=LIMITS['available'] and sample['disk']>=LIMITS['disk']
        samples_count+=len(samples);peak=max(peak,run['peak_rss'])
        assert len(result['outputs'])==36
        control=next(row for row in controls['jobs'] if row['job']==job);assert control['name']==request['name']
        boundaries=[];output_pins=[]
        for index,(row,desc) in enumerate(zip(result['outputs'],spec['outputs'],strict=True),5):
            assert row['index']==index and row['name']==desc['name'] and row['shape']==desc['shape']
            path=folder/row['file'];assert pin(path)==row['pin'] and path.stat().st_size==math.prod(desc['shape'])*4
            actual=np.fromfile(path,dtype='<f4').reshape(desc['shape']);output_pins.append(row['pin'])
            references={}
            for engine in ['numpy','ort']:
                reference=request['references'][engine][index-5]
                assert reference['name']==desc['name'] and reference['shape']==desc['shape']
                value=np.fromfile(ROOT/reference['file'],dtype='<f8').reshape(desc['shape'])
                references[engine]=checked_metric(actual,value)
            original=control['boundaries'][index];assert original['name']==desc['name']
            boundaries.append(dict(index=index,name=desc['name'],references=references,control_references=original['references']))
        direct={kind:checked_metric(actual,source(desc)) for kind,desc in request['baselines'].items()}
        pins[job['id']]=output_pins
        rows.append(dict(job=job,name=request['name'],boundaries=boundaries,original_fp32_final=direct,seconds=result['seconds']))
    for mode in MODES:assert pins['00-'+mode]==pins['03-'+mode]
    screens={};screen_rows=[]
    for mode in MODES:
        outcomes=[]
        for row in rows:
            if row['job']['mode']!=mode:continue
            for engine in ['numpy','ort']:
                final=row['boundaries'][-1];a=final['references'][engine];b=final['control_references'][engine]
                ok=a['max_scaled']<=.5*b['max_scaled'] and a['failed_values']<=b['failed_values']
                outcomes.append(ok)
                screen_rows.append(dict(job=row['job']['id'],reference=engine,passed=ok,maximum_ratio=a['max_scaled']/b['max_scaled']))
        screens[mode]=all(outcomes)
    analysis=dict(protocol=PROTOCOL,structural_passed=True,screens=screens,screen_rows=screen_rows,jobs=rows,arrays=288,
        comparisons=8*(36*2+2+1),metric_checks=8*(36*2+2+1)*4,scalar_coordinates=scalar_count,samples=samples_count,
        peak_rss=peak,births=births,manifest=pin(BASE/'manifest.json'),source_revision=spec['source_revision'],
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    lines=['# Whisper stem-state intervention — 2026-09-21','',
        'This diagnostic replaces the state after the first 14 original encoder nodes with its saved '
        'NumPy/SciPy float64 reference, rounded once to float32. It executes the remaining 1,545 nodes '
        'with the same two arithmetic modes as the prior experiment. The completed original controls '
        'are reused without replay. This is an intervention in an unfused interpreter, not a managed '
        'product implementation or a performance measurement.','',
        '| Request | Suffix mode | Control final max | Corrected-stem final max | Control failed values | Corrected-stem failed values |',
        '|---|---|---:|---:|---:|---:|']
    for row in rows:
        final=row['boundaries'][-1];a=final['references']['numpy'];b=final['control_references']['numpy']
        lines.append(f"| {row['job']['request']}: {row['name']} | {row['job']['mode']} | {b['max_scaled']:.9g} | {a['max_scaled']:.9g} | {b['failed_values']:,} | {a['failed_values']:,} |")
    lines+=['','The prospective stem-intervention screen requires at least a halving of final maximum '
        'error for every selected case against both float64 reference routes, without increasing failed '
        'value counts. Results by suffix mode: '+', '.join(f"**{m}: {'passes' if screens[m] else 'fails'}**" for m in MODES)+'.','',
        '| Suffix mode | Failed boundary/reference comparisons | Total boundary/reference comparisons |',
        '|---|---:|---:|']
    for mode in MODES:
        comparisons=[b['references'][e] for r in rows if r['job']['mode']==mode for b in r['boundaries'] for e in ['numpy','ort']]
        lines.append(f"| {mode} | {sum(v['failed_values']>0 for v in comparisons)} | {len(comparisons)} |")
    lines+=['','All comparisons use `abs(actual-reference)/max(1,abs(reference))`, with failures above '
        '`1e-4`. Every value and padded frame is retained. The complete observations include both reference '
        'routes, all 36 suffix boundaries, control errors and original managed/native final comparisons. '
        'A positive intervention result only nominates stem precision as a future implementation target; '
        'it cannot attribute the effect to one convolution or activation, establish a managed-kernel '
        'benefit, or waive the existing full-encoder/native gates.','',
        'Inputs are the same three previously selected natural recordings and first-case repeat, with '
        'managed features. Both independently computed stem references agree within 7.345e-16. Their '
        'float32 casts differ at one near-zero coordinate in the first case by 4.646e-16; the other two '
        'cases cast identically. The declared NumPy reference is used throughout. Repeated intervention '
        'inputs and all resulting arrays match exact bits.','',
        f'All eight workers and the supervisor are terminal. All 288 output arrays, {samples_count:,} resource '
        f"samples, {analysis['metric_checks']:,} independent metric checks and {scalar_count:,} math.fsum dot "
        f'coordinates pass structural verification. Peak sampled RSS: {peak:,} bytes. CPU2 and one numerical '
        'library thread are verified. No production code, default or timing result changes.','',
        '[Original matrix-precision controls](../matmul-precision/results-20260921.md); '
        '[complete observations](observations-20260921.json). '
        f"Frozen source `{spec['source_revision']}`; manifest SHA256 `{analysis['manifest']['sha256']}`. "
        'Artifact: `artifacts/whisper-stem-intervention-20260921`.','']
    report=Path(__file__).with_name('results-20260921.md');observations=report.with_name('observations-20260921.json')
    assert not report.exists() and not observations.exists()
    write(BASE/'analysis.json',analysis);write(observations,analysis)
    with report.open('x',encoding='utf8') as stream:stream.write('\n'.join(lines))
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    write(BASE/'closed.json',dict(structural_passed=True,screens=screens,files=files,births=births,reports={rel(report):pin(report),rel(observations):pin(observations)}))
    print(json.dumps(dict(closed=True,screens=screens,arrays=288,receipt=pin(BASE/'closed.json'))))


if __name__=='__main__':main()
