"""Independently reproduce scalar fingerprints, transition entries and complete timing screens."""
from pathlib import Path
import argparse,json,math,statistics
from generate import pin,CORE


def fixture(value):
    h=1469598103934665603;transitions=[]
    def number(v):
        nonlocal h
        h=((h^(v&((1<<64)-1)))*1099511628211)&((1<<64)-1)
    def integer(v):number(v&0xffffffff)
    def string(units):
        before=h
        if units is None:number(0x9E3779B97F4A7C15)
        else:
            assert isinstance(units,list) and all(type(c) is int and 0<=c<=65535 for c in units)
            integer(len(units))
            for c in units:number(c)
        transitions.append(dict(before=str(before),value=units,after=str(h)))
    integer(len(value['nodes']))
    for node in value['nodes']:
        string(node['Name']);integer(node['op']);string(node['Domain']);string(node['OpTypeName'])
        for key in ['Inputs','Outputs']:
            values=node[key]
            integer(-1 if values is None else len(values))
            if values is not None:
                for item in values:string(item)
    for key in ['inputs','initializers','input_descriptions','output_descriptions']:
        integer(len(value[key]))
        for item in value[key]:string(item)
    assert int(value['fingerprint'])==h-(1<<64 if h>=1<<63 else 0)
    assert value['transitions']==transitions
    return len(transitions)


def worker(folder,generation,binaries,timing):
    value=json.loads((folder/'result.json').read_text());fixtures=json.loads((folder/'fixtures.json').read_text())
    assert value['schema']==1 and value['passed'] is True and value['timing'] is timing
    assert value['core_sha256']==CORE and value['model_sha256']==generation['model']['sha256']
    assert value['probe_sha256']==binaries['Probe.dll']['sha256'] and value['fixtures_sha256']==pin(folder/'fixtures.json')['sha256']
    assert value['settings']==[] and value['affinity']==4 and value['cycle_refusals']==4 and value['concurrent_checks']==128
    assert value['runtime']==('.NET 10.0.8' if timing else '.NET 10.0.12')
    assert value['checks']==40*(1+160*2+3)+7+1+2*347+1 and value['nodes']==347 and value['initializers']==270
    assert len(fixtures)==40 and [r['name'] for r in fixtures]==['flat-'+str(i) for i in range(40)]
    transitions=sum(fixture(r) for r in fixtures)
    assert value['cache_graphs']==1 and value['cache_entries']>0 and value['cache_struct_bytes']==24*value['cache_entries']<2*1024**2
    assert value['training_ticks']>0 and value['load_ticks']>0 and value['training_allocated_bytes']>=value['cache_struct_bytes']
    warm,cycles,repeats=(32,64,256) if timing else (2,4,8)
    assert (value['warmup_cycles'],value['measured_cycles'],value['repeats'])==(warm,cycles,repeats)
    assert type(value['frequency']) is int and value['frequency']>0 and len(value['samples'])==4*cycles
    for row,(cycle,position) in zip(value['samples'],((c,p) for c in range(cycles) for p in range(4)),strict=True):
        assert (row['cycle'],row['position'],row['variant'])==(cycle,position,(value['visit']+warm+cycle+position)%4)
        assert type(row['ticks']) is int and row['ticks']>0 and row['bytes']==0
        assert len(row['gc'])==3 and all(type(v) is int and v>=0 for v in row['gc'])
    return value,transitions


def resources(folder):
    state=json.loads((folder/'identity.json').read_text())
    assert state['complete'] is True and state['code']==0 and 'error' not in state
    assert state['limits']==dict(seconds=120,rss=2*1024**3,available=1024**3)
    previous_end=state['started'];births={state['supervisor']['pid']:state['supervisor']['birth']};peak=0;minimum=math.inf
    for index,row in enumerate(state['runs']):
        assert row['visit']==index and row['code']==0 and 0<row['seconds']<120
        assert previous_end<=row['started']<=row['ended']<=state['ended'];previous_end=row['ended']
        samples=[json.loads(line) for line in (folder/f'v{index}/samples.jsonl').read_text().splitlines()]
        assert len(samples)==row['samples'] and len(samples)>1
        previous=-1;seen={};worker_peak=0
        for sample in samples:
            assert previous<sample['seconds']<row['seconds'];previous=sample['seconds']
            assert sample['available']>=1024**3;minimum=min(minimum,sample['available'])
            assert len({m['pid'] for m in sample['members']})==len(sample['members'])
            for m in sample['members']:
                assert m['affinity']==[2] and m['rss']>=0 and m['birth']==row['members'][str(m['pid'])]>=row['child']['birth']
                seen[str(m['pid'])]=m['birth']
            worker_peak=max(worker_peak,sum(m['rss'] for m in sample['members']))
        assert worker_peak==row['peak_rss']<2*1024**3 and seen==row['members']
        assert row['members'][str(row['child']['pid'])]==row['child']['birth']
        births.update({int(p):b for p,b in row['members'].items()});peak=max(peak,worker_peak)
    return dict(peak_rss=peak,minimum_available=minimum,births=[dict(pid=p,birth=b) for p,b in births.items()])


def timing(values):
    assert len(values)==4 and [v['visit'] for v in values]==list(range(4))
    visits=[]
    for value in values:
        means=[statistics.mean(r['ticks']/value['frequency']/value['repeats'] for r in value['samples'] if r['variant']==i) for i in range(4)]
        visits.append(dict(visit=value['visit'],means=means,cached_actual=means[3]/means[0],copy_b_a=means[2]/means[1]))
    means=[statistics.mean(v['means'][i] for v in visits) for i in range(4)]
    gates=dict(duplicate_aggregate=.98<=means[2]/means[1]<=1.02,
        duplicate_workers=all(.95<=r['copy_b_a']<=1.05 for r in visits),
        actual_copy=all(.9<=means[i]/means[0]<=1.1 for i in (1,2)),
        gain=means[3]/means[0]<=.75,worker_regression=all(r['cached_actual']<=1.01 for r in visits))
    return dict(passed=all(gates.values()),gates=gates,mean_seconds=means,cached_actual=means[3]/means[0],visits=visits,
        scope='Prospective component-only nomination screens; no complete-model latency claim or A/A calibration repair')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--process',default='timing-process');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    base=a.artifact.resolve();generation=json.loads((base/'generation.json').read_text());folder=base/a.process
    state=json.loads((folder/'identity.json').read_text());limits=resources(folder)
    for name,wanted in generation['files'].items():assert pin(base/name)==wanted,name
    for name,wanted in state['binaries'].items():assert pin(base/'bin'/name)==wanted,name
    values=[];counts=[]
    for r in state['runs']:
        value,count=worker(folder/f"v{r['visit']}/output",generation,state['binaries'],state['mode']=='timing');values.append(value);counts.append(count)
    result=dict(passed=True,mode=state['mode'],resources=limits,checks=sum(v['checks'] for v in values),independent_transitions=sum(counts),
        generation=pin(base/'generation.json'),process=pin(folder/'identity.json'),auditor=pin(Path(__file__)),
        timing=timing(values) if state['mode']=='timing' else None)
    with a.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
