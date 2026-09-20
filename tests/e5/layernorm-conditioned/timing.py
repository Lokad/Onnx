"""Validate exact fixed conditioning cutoff and every longer fixed timing cycle."""
import statistics
from common import BANKS,CORE,VARIANTS,order

def conditioning(case,frequency):
    warm=case['warmup'];assert len(warm)>0 and len(warm)%4==0
    assert case['conditioning_seconds']==3
    assert all(type(s['ticks']) is int and s['ticks']>0 for s in warm)
    total=sum(s['ticks'] for s in warm)
    assert case['conditioning_ticks']==total and total>=3*frequency
    assert total-sum(s['ticks'] for s in warm[-4:])<3*frequency,'Conditioning continued after its first complete qualifying cycle'
    return len(warm)//4

def inspect_timing(value,visit,probe,descriptions):
    assert value['schema']==1 and value['protocol']=='layernorm-conditioned-bank-v2' and value['visit']==visit
    identity=value['identity'];assert identity['mode']=='run' and identity['core_sha256']==CORE and identity['probe_sha256']==probe['sha256']
    assert identity['runtime']=='10.0.8' and identity['affinity']==4 and identity['vector_width']==8 and identity['settings']=={}
    assert identity['vector512_hardware'] is True and identity['avx512'] is True and value['frequency']==1000000000
    assert value['case_order']==order(visit) and [c['name'] for c in value['results']]==[d['name'] for d in BANKS]
    rows=[]
    for definition,case,description in zip(BANKS,value['results'],descriptions):
        assert case['nodes']==25 and case['values']==25*definition['rows']*definition['width'] and case['held_unchanged'] is True
        for key,want in description.items():
            if want is not None:assert case[key]==want,(case['name'],key)
        assert len(case['output_sha256'])==64
        cycles=conditioning(case,value['frequency'])
        for label,count in [('first',1),('warmup',cycles),('measured',96)]:
            samples=case[label];assert len(samples)==count*4
            for index,sample in enumerate(samples):
                cycle,position=divmod(index,4);variant=position if label=='first' else (visit+cycle+position)%4
                assert (sample['cycle'],sample['position'],sample['variant'],sample['repeats'])==(-1 if label=='first' else cycle,position,VARIANTS[variant],1 if label=='first' else definition['repeats'])
                assert type(sample['ticks']) is int and sample['ticks']>0 and type(sample['allocated']) is int and sample['allocated']>=0
                assert len(sample['gc'])==3 and all(type(v)is int and v>=0 for v in sample['gc'])
                assert sample['output_sha256']==case['output_sha256']
        for variant in VARIANTS:
            samples=[s for s in case['measured'] if s['variant']==variant];assert len(samples)==96
            times=[s['ticks']/s['repeats']/value['frequency']*1000 for s in samples]
            rows.append(dict(case=case['name'],visit=visit,variant=variant,mean_ms=statistics.mean(times),median_ms=statistics.median(times),min_ms=min(times),max_ms=max(times),
                ticks=sum(s['ticks'] for s in samples),banks=sum(s['repeats'] for s in samples),allocated=sum(s['allocated'] for s in samples),gc=[sum(s['gc'][i] for s in samples) for i in range(3)]))
    return rows
