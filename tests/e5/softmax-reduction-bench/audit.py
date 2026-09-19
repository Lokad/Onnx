"""Validate fixed long-batch control/comparison evidence without changing its gates."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,statistics,struct

SHAPES=['8','30','pad128','128','512','pad512']
MODES=['actual','copied','duplicate','probe']
WIDTHS=[8,30,128,128,512,512]
ITERATIONS=[32768,4096,256,256,32,32]
CORE='187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4'

def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sequence(order):
    values=[(i+order%4)%4 for i in range(4)]
    return values if order<4 else values[::-1]

def validate_worker(value,phase,order):
    require(phase in ('control','compare'),'Unknown phase')
    require(value['schema']==2 and value['protocol']=='softmax-reduction-batches-v1' and value['phase']==phase and value['order']==order and value['sequence']==sequence(order),'Worker schedule differs')
    require(not value['checkOnly'] and value['width']==8 and value['affinity']==4 and value['frequency']==1_000_000_000 and value['runtime']=='.NET 10.0.8','Worker environment differs')
    require(value['core_sha256']==CORE and value['maximumCases']==109488 and value['tensorCases']==1728 and value['inPlaceCases']==3456 and value['refusals']==3 and 0<=value['maximumDoubleError']<=1e-6,'Correctness coverage differs')
    require([(r['name'],r['mode']) for r in value['records']]==[(n,m) for n in SHAPES for m in MODES],'Timing coverage differs')
    for row in value['records']:
        i=SHAPES.index(row['name']);columns=WIDTHS[i];active=30 if row['name'].startswith('pad') else columns
        require((row['rows'],row['columns'],row['iterations'])==(12*columns,columns,ITERATIONS[i]),'Geometry differs')
        expected='SoftmaxReduction.Kernels.Reduced' if phase=='compare' and row['mode']=='probe' else 'SoftmaxReduction.Kernels.Original'
        if row['mode']=='actual':require(row['implementation'].startswith('Lokad.Onnx.Tensor`1[') and row['implementation'].endswith('.SoftmaxMaskedFloatSpanPtr'),'Actual implementation differs')
        else:require(row['implementation']==expected,'Implementation alias differs')
        mask=struct.pack('<f',0)*active+struct.pack('<I',0xff7fffff)*(columns-active)
        require(row['mask_sha256']==hashlib.sha256(mask).hexdigest(),'Mask differs')
        require(type(row['conditioning_calls']) is int and row['conditioning_calls']>=row['iterations'] and row['conditioning_calls']%row['iterations']==0,'Conditioning differs')
        require(len(row['samples'])==9,'Sample coverage differs')
        for sample in row['samples']:
            require(all(type(sample[k]) is int and sample[k]>0 for k in ('ticks','thread_ns','process_ns')),'Invalid timing counter')
            require(len(sample['gc'])==3 and all(type(g) is int and g>=0 for g in sample['gc']),'Invalid GC counter')
        family=[r for r in value['records'] if r['name']==row['name']]
        require(all((r['input_sha256'],r['mask_sha256'],r['output_sha256'])==(row['input_sha256'],row['mask_sha256'],row['output_sha256']) for r in family),'Cross-mode values differ')

def evaluate(workers,phase,telemetry):
    require(len(workers)==8,'Worker count differs')
    for i,w in enumerate(workers):validate_worker(w,phase,i)
    results={};control_passed=candidate_passed=True
    for name in SHAPES:
        means={m:[statistics.mean(s['ticks']/r['iterations']/1e6 for s in r['samples']) for w in workers for r in w['records'] if r['name']==name and r['mode']==m] for m in MODES}
        aggregate={m:statistics.mean(v) for m,v in means.items()}
        ratios={m:[a/b for a,b in zip(means[m],means['copied'],strict=True)] for m in ('duplicate','probe')}
        controls={m:dict(aggregate_within_1_percent=abs(aggregate[m]/aggregate['copied']-1)<=.01,every_worker_within_2_percent=all(abs(v-1)<=.02 for v in ratios[m])) for m in (('duplicate','probe') if phase=='control' else ('duplicate',))}
        controls['copy_actual']=dict(aggregate_within_3_percent=abs(aggregate['copied']/aggregate['actual']-1)<=.03)
        candidate={}
        if phase=='compare':
            if name in ('30','128'):
                candidate=dict(aggregate_copy_gain_at_least_3_percent=aggregate['probe']/aggregate['copied']<=.97,
                               aggregate_actual_gain_at_least_3_percent=aggregate['probe']/aggregate['actual']<=.97,
                               every_worker_regression_at_most_2_percent=all(v<=1.02 for v in ratios['probe']))
            else:
                candidate=dict(aggregate_regression_at_most_2_percent=aggregate['probe']/aggregate['copied']<=1.02,every_worker_regression_at_most_5_percent=all(v<=1.05 for v in ratios['probe']))
        control_passed &= all(all(c.values()) for c in controls.values());candidate_passed &= all(candidate.values())
        results[name]=dict(mean_ms=aggregate,worker_means_ms=means,worker_ratios_to_copy=ratios,control_criteria=controls,candidate_criteria=candidate)
    samples=[s for w in workers for r in w['records'] for s in r['samples']]
    health=dict(every_batch_at_least_20_ms=all(s['ticks']>=20_000_000 for s in samples),no_measured_gc=not any(any(s['gc']) for s in samples),foreign_cpu_at_most_2_percent=telemetry['maximum_foreign_cpu_fraction']<=.02,steal_at_most_half_percent=telemetry['maximum_steal_fraction']<=.005)
    return dict(passed=bool(control_passed and candidate_passed and all(health.values())),control_passed=bool(control_passed),candidate_passed=bool(candidate_passed) if phase=='compare' else None,health=health,minimum_batch_ms=min(s['ticks'] for s in samples)/1e6,results=results)

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--phase',choices=['control','compare'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'Existing audit output');base=a.artifact;collected=base/('collected-'+a.phase);out=collected/('result-'+a.phase)
    bundle=read(base/'payload/bundle.json');identity=read(out/'identity.json');collection=read(collected/('collection-'+a.phase+'.json'))
    require(identity['phase']==a.phase and identity['complete'] and len(identity['runs'])==8 and (collected/('complete-'+a.phase+'.txt')).read_text().strip()=='0','Incomplete campaign')
    require(collection['phase']==a.phase and collection['supervisor']==dict(pid=identity['supervisor'],start=identity['supervisor_start']) and collection['terminal_workers']==[dict(pid=r['pid'],start=r['start_identity'],code=r['code']) for r in identity['runs']],'Terminal identities differ')
    for name,pin in collection['files'].items():require(sha(collected/name)==pin['sha256'] and (collected/name).stat().st_size==pin['bytes'],'Collection differs: '+name)
    require(sha(collected/'bundle.json')==sha(base/'payload/bundle.json'),'Bundle manifest differs')
    for name,pin in bundle['files'].items():require(sha(base/'payload'/name)==sha(collected/name)==pin['sha256'],'Frozen payload differs: '+name)
    require(sha(__file__)==bundle['files']['audit.py']['sha256'],'Auditor differs from frozen protocol')
    if a.phase=='compare':
        gate=read(base/'control-audit.json');require(gate['phase']=='control' and gate['passed'] and gate['integrity_passed'],'Prior control gate failed')
        require(sha(base/'control-audit.json')==sha(collected/'control-audit.json')==identity['control_audit_sha256'],'Prior audit differs')
        require(gate['identity_sha256']==sha(collected/'result-control/identity.json') and gate['bundle_manifest_sha256']==sha(collected/'bundle.json'),'Prior control identity differs')
    else:require(identity['control_audit_sha256'] is None,'Unexpected control gate')
    spec=importlib.util.spec_from_file_location('frozen_processes',collected/'campaign_processes.py');account=importlib.util.module_from_spec(spec);spec.loader.exec_module(account)
    workers=[];telemetry=dict(maximum_foreign_cpu_fraction=0.,maximum_steal_fraction=0.,maximum_rss=0)
    for i,row in enumerate(identity['runs']):
        require(row['index']==row['order']==i and row['name']=='worker-'+str(i) and row['flags']=={} and row['code']==0 and row['seconds']<180 and row['samples'],'Worker identity/bounds differ')
        directory=out/row['name'];w=read(directory/'samples.json');validate_worker(w,a.phase,i);require(w['probe_sha256']==bundle['files']['bin/Probe.dll']['sha256'],'Probe differs');workers.append(w)
        for r in w['records']:
            require({k:r[k] for k in ('input_sha256','mask_sha256','output_sha256')}==bundle['expected_workloads'][r['name']],'Pinned workload differs')
        observed=account.foreign_fraction(read(directory/'pre.json'),read(directory/'post.json'),identity['supervisor']);require(observed==row['accounting'],'Accounting differs')
        telemetry['maximum_foreign_cpu_fraction']=max(telemetry['maximum_foreign_cpu_fraction'],observed['foreign_cpu_fraction'])
        before=[int(v) for v in (directory/'cpu-pre.txt').read_text().splitlines()[0].split()[1:]];after=[int(v) for v in (directory/'cpu-post.txt').read_text().splitlines()[0].split()[1:]]
        delta=[y-x for x,y in zip(before,after,strict=True)];require(all(v>=0 for v in delta) and sum(delta[:8])>0,'CPU counters invalid')
        telemetry['maximum_steal_fraction']=max(telemetry['maximum_steal_fraction'],delta[7]/sum(delta[:8]))
        require(row['members'][str(row['pid'])]['start']==row['start_identity'],'Worker birth missing')
        for sample in row['samples']:
            for member in sample['members']:require(member['affinity']=='2' and row['members'][str(member['pid'])]['start']==member['start'],'Process birth/affinity differs')
            rss=sum(m['rss'] for m in sample['members']);require(rss<1024**3,'RSS guard exceeded');telemetry['maximum_rss']=max(telemetry['maximum_rss'],rss)
    verdict=evaluate(workers,a.phase,telemetry)
    result=dict(schema=1,phase=a.phase,integrity_passed=True,**verdict,telemetry=telemetry,workers=8,measured_batches=1728,scope='Empirical kernel protocol and mechanism screen, not calibrated whole-model confidence',
        audit_sha256=sha(__file__),bundle_manifest_sha256=sha(collected/'bundle.json'),identity_sha256=sha(out/'identity.json'),collection_sha256=sha(collected/('collection-'+a.phase+'.json')),archive_sha256=sha(base/(a.phase+'-results.tar.gz')))
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(a.phase,'passed',result['passed'],'control',result['control_passed'],'candidate',result['candidate_passed'],'health',result['health'])
    for name,r in result['results'].items():print(name,r['mean_ms'],r['control_criteria'],r['candidate_criteria'])

if __name__=='__main__':main()
