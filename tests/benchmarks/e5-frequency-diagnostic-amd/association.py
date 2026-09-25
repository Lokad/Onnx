"""Retain all intervals; associate only intervals fully inside fixed call blocks."""
from decimal import Decimal
from counter import EVENTS


def associate(clocks, rows, bounds):
    assert len(clocks)==780 and all(c['frequency']==1_000_000_000 for c in clocks)
    blocks=[]
    for first in range(0,780,60):
        group=clocks[first:first+60]
        blocks.append(dict(first=first,last=first+59,warmup=first<600,start_ns=group[0]['start'],
            end_ns=group[-1]['end'],wall_ms=sum(c['ticks'] for c in group)/60/1e6,
            intervals=[],covered_ns=0,totals={e:Decimal(0) for e in EVENTS}))
    retained=[]
    prior=0
    for index,row in enumerate(rows):
        elapsed=row['elapsed_ns'];assert elapsed>prior
        start_low=bounds['lower_ns']+prior;start_high=bounds['upper_ns']+prior
        end_low=bounds['lower_ns']+elapsed;end_high=bounds['upper_ns']+elapsed
        matches=[b for b in blocks if start_low>=b['start_ns'] and end_high<=b['end_ns']]
        assert len(matches)<=1
        reason='initial enable interval' if index==0 else 'outside or crossing a block boundary'
        selected=None
        if matches and index>0:
            block,=matches
            for event in EVENTS:
                value=row['events'][event]
                assert value['count'] is not None and Decimal(value['running_percent'])>=Decimal('99.9')
                count=Decimal(value['count']);assert count>=0
                if event in EVENTS[:3]:assert count>0
                block['totals'][event]+=count
            block['intervals'].append(index);block['covered_ns']+=elapsed-prior
            reason='wholly within block at both epoch bounds';selected=block['first']
        retained.append(dict(index=index,previous_elapsed_ns=prior,elapsed_ns=elapsed,
            start_lower_ns=start_low,start_upper_ns=start_high,end_lower_ns=end_low,end_upper_ns=end_high,
            block_first=selected,reason=reason,events=row['events']))
        prior=elapsed
    for block in blocks:
        block['coverage']=block['covered_ns']/(block['end_ns']-block['start_ns'])
        assert .85<=block['coverage']<=1,('Insufficient counter coverage',block['first'],block['coverage'])
        block['aperf_mperf']=float(block['totals'][EVENTS[0]]/block['totals'][EVENTS[1]])
        block['mperf_tsc']=float(block['totals'][EVENTS[1]]/block['totals'][EVENTS[2]])
        block['totals']={name:str(count) for name,count in block['totals'].items()}
    assert sum(len(b['intervals']) for b in blocks)==sum(r['block_first'] is not None for r in retained)
    return dict(blocks=blocks,intervals=retained)
