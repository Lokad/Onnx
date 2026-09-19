"""Audit complete fixed zero-block prototype and apply the prospective mechanism screen."""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,statistics,struct

SHAPES=['8','30','pad128','128','512','pad512'];MODES=['actual','copied','skip','adaptive']
WIDTHS=[8,30,128,128,512,512];ITERATIONS=[128,128,16,16,2,2]

def require(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def validate_worker(value,order):
    require(value['schema']==1 and value['order']==order and not value['checkOnly'] and value['width']==8,'Worker identity differs')
    require(value['expValues']==2503712 and value['tensorCases']==918 and value['refusals']==4 and 0<=value['maximumDoubleError']<=1e-6,'Correctness coverage differs')
    require(value['core_sha256']=='7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9','Core differs')
    require(value['frequency']==1_000_000_000 and value['runtime']=='.NET 10.0.8','Runtime/clock differs')
    require([(r['name'],r['mode']) for r in value['records']]==[(n,m) for n in SHAPES for m in MODES],'Timing coverage differs')
    for row in value['records']:
        i=SHAPES.index(row['name']);columns=WIDTHS[i];active=30 if row['name'].startswith('pad') else columns
        require((row['rows'],row['columns'],row['iterations'])==(12*columns,columns,ITERATIONS[i]),'Geometry/iterations differ')
        mask=struct.pack('<f',0)*active+struct.pack('<I',0xff7fffff)*(columns-active)
        require(row['mask_sha256']==hashlib.sha256(mask).hexdigest(),'Mask differs')
        require(type(row['conditioning_calls']) is int and row['conditioning_calls']>0,'Conditioning missing')
        require(len(row['samples'])==9,'Sample count differs')
        for s in row['samples']:
            require(all(type(s[k]) is int and s[k]>0 for k in ('ticks','thread_ns','process_ns')),'Invalid wall/CPU sample')
            require(len(s['gc'])==3 and all(type(g) is int and g>=0 for g in s['gc']),'Invalid GC counters')
        family=[r for r in value['records'] if r['name']==row['name']]
        require(all((r['input_sha256'],r['mask_sha256'],r['output_sha256'])==(row['input_sha256'],row['mask_sha256'],row['output_sha256']) for r in family),'Cross-mode input/output differs')

def screen(workers):
    results={};passed=True
    for name in SHAPES:
        means={mode:[statistics.mean(s['ticks']/r['iterations']/1e6 for s in r['samples']) for w in workers for r in w['records'] if r['name']==name and r['mode']==mode] for mode in MODES}
        aggregate={k:statistics.mean(v) for k,v in means.items()}
        ratios={m:[a/b for a,b in zip(means[m],means['copied'],strict=True)] for m in ('skip','adaptive')}
        row=dict(mean_ms=aggregate,worker_means_ms=means,worker_ratios_to_copy=ratios,adaptive_to_actual=aggregate['adaptive']/aggregate['actual'],copy_to_actual=aggregate['copied']/aggregate['actual'])
        if name=='pad128':
            row['criteria']=dict(every_adaptive_visit_at_least_15_percent=all(v<=.85 for v in ratios['adaptive']),aggregate_actual_gain_at_least_15_percent=row['adaptive_to_actual']<=.85,
                copy_actual_control_within_3_percent=abs(row['copy_to_actual']-1)<=.03)
        elif not name.startswith('pad'):
            row['criteria']=dict(aggregate_regression_at_most_2_percent=aggregate['adaptive']/aggregate['copied']<=1.02,every_visit_regression_at_most_5_percent=all(v<=1.05 for v in ratios['adaptive']))
        else:row['criteria']={}
        passed &= all(row['criteria'].values());results[name]=row
    return passed,results

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();base=a.artifact
    require(not a.output.exists(),'Existing audit output');payload=base/'payload';collected=base/'collected';out=collected/'result';bundle=read(payload/'bundle.json');identity=read(out/'identity.json')
    collection=read(collected/'collection.json')
    require(collection['terminal_supervisor']['pid']==identity['supervisor'] and len(collection['terminal_workers'])==9,'Terminal receipt differs')
    require([(r['pid'],r['start'],r['code']) for r in collection['terminal_workers']]==[(r['pid'],r['start_identity'],r['code']) for r in identity['runs']],'Terminal worker identities differ')
    for name,pin in collection['files'].items():
        require(sha(collected/name)==pin['sha256'] and (collected/name).stat().st_size==pin['bytes'],'Collection differs: '+name)
    require((collected/'complete.txt').read_text().strip()=='0' and identity['complete'] and len(identity['runs'])==9,'Campaign incomplete')
    for name,pin in bundle['files'].items():
        require(sha(payload/name)==sha(collected/name)==pin['sha256'] and (collected/name).stat().st_size==pin['bytes'],'Payload differs: '+name)
    helper=collected/'campaign_processes.py';spec=importlib.util.spec_from_file_location('frozen_processes',helper);account=importlib.util.module_from_spec(spec);spec.loader.exec_module(account)
    workers=[];max_foreign=max_steal=0.;max_rss=0;gc_samples=0
    for index,row in enumerate(identity['runs']):
        expected='disassembly' if index==8 else 'worker-'+str(index)
        require(row['index']==index and row['name']==expected and row['order']==(0 if index==8 else index) and row['disassembly']==(index==8),'Schedule differs')
        require(row['code']==0 and row['seconds']<180 and len(row['samples'])>0,'Worker failure/bound')
        directory=out/expected;value=read(directory/'samples.json');validate_worker(value,row['order']);require(value['probe_sha256']==bundle['files']['bin/Probe.dll']['sha256'],'Probe differs')
        if index<8:require(row['flags']=={},'Normal settings changed');workers.append(value)
        pre=read(directory/'pre.json');post=read(directory/'post.json');observed=account.foreign_fraction(pre,post,identity['supervisor'])
        require(observed==row['accounting'],'Accounting differs')
        max_foreign=max(max_foreign,observed['foreign_cpu_fraction'])
        before=[int(v) for v in (directory/'cpu-pre.txt').read_text().splitlines()[0].split()[1:]]
        after=[int(v) for v in (directory/'cpu-post.txt').read_text().splitlines()[0].split()[1:]]
        delta=[y-x for x,y in zip(before,after,strict=True)];require(all(v>=0 for v in delta),'CPU counters moved backwards');max_steal=max(max_steal,delta[7]/sum(delta[:8]))
        for sample in row['samples']:
            for member in sample['members']:
                require(member['affinity']=='2' and row['members'][str(member['pid'])]['start']==member['start'],'Process identity/affinity differs')
            rss=sum(m['rss'] for m in sample['members']);max_rss=max(max_rss,rss);require(rss<1024**3,'RSS guard exceeded')
        require(str(row['pid']) in row['members'] and row['members'][str(row['pid'])]['start']==row['start_identity'],'Worker birth missing')
        gc_samples+=sum(any(s['gc']) for r in value['records'] for s in r['samples'])
    for i in range(24):
        identities={(w['records'][i]['input_sha256'],w['records'][i]['mask_sha256'],w['records'][i]['output_sha256']) for w in workers};require(len(identities)==1,'Cross-worker values differ')
    passed,results=screen(workers)
    code=out/'disassembly/codegen.txt';require(code.exists() and code.stat().st_size>0,'Missing disassembly')
    output=dict(schema=1,correctness_and_integrity_passed=True,prospective_timing_screen_passed=passed,scope='Kernel mechanism screen; no full-model or calibrated confidence claim',
        workers=8,codegen_workers=1,measured_samples=8*24*9,all_worker_checks=dict(exponential_values=9*2503712,tensor_cases=9*918,refusals=9*4),
        maximum_double_error=max(w['maximumDoubleError'] for w in workers),maximum_foreign_cpu_fraction=max_foreign,maximum_steal_fraction=max_steal,maximum_rss=max_rss,
        measured_batches_with_gc_including_codegen=gc_samples,results=results,audit_sha256=sha(__file__),bundle_sha256=sha(base/'bundle.tar.gz'),archive_sha256=sha(base/'results.tar.gz'),
        identity_sha256=sha(out/'identity.json'),collection_sha256=sha(collected/'collection.json'),disassembly_sha256=sha(code))
    with a.output.open('x',encoding='utf-8') as f:json.dump(output,f,indent=2);f.write('\n')
    print('screen',passed,'max foreign',max_foreign,'max RSS',max_rss)
    for name,r in results.items():print(name,r['mean_ms'],r['criteria'])

if __name__=='__main__':main()
