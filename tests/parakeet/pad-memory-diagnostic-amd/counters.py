"""Independent counter-bracket checks and descriptive, untrimmed associations."""
from collections import defaultdict
from math import prod
import statistics


def inspect(clock, frequency, empty=False):
    c=clock['counters']
    assert all(type(v) is int for v in c.values()) and frequency>0
    assert c['begin']<=c['start']<=c['stop']<=c['end']
    assert c['end']>c['begin'] and c['stop']-c['start']==clock['ticks']
    if not empty:assert clock['ticks']>0
    for left,right in [('allocatedBefore','allocatedAfter'),('gc0','after0'),('gc1','after1'),('gc2','after2'),
            ('userBeforeUs','userAfterUs'),('systemBeforeUs','systemAfterUs'),
            ('minorBefore','minorAfter'),('majorBefore','majorAfter'),
            ('voluntaryBefore','voluntaryAfter'),('involuntaryBefore','involuntaryAfter')]:
        assert 0<=c[left]<=c[right],(left,right)
    return dict(pad_us=clock['ticks']*1e6/frequency,
        bracket_us=(c['end']-c['begin'])*1e6/frequency,
        allocated=c['allocatedAfter']-c['allocatedBefore'],
        minor=c['minorAfter']-c['minorBefore'],major=c['majorAfter']-c['majorBefore'],
        user_us=c['userAfterUs']-c['userBeforeUs'],system_us=c['systemAfterUs']-c['systemBeforeUs'],
        voluntary=c['voluntaryAfter']-c['voluntaryBefore'],
        involuntary=c['involuntaryAfter']-c['involuntaryBefore'],
        collection=any(c[f'after{g}']!=c[f'gc{g}'] for g in range(3)))


def summarize(rows):
    assert rows
    value=dict(calls=len(rows),minor=sum(r['minor'] for r in rows),major=sum(r['major'] for r in rows),
        collection_calls=sum(r['collection'] for r in rows),
        switch_calls=sum(r['voluntary']>0 or r['involuntary']>0 for r in rows))
    for key in ('pad_us','bracket_us','allocated','minor','user_us','system_us'):
        value[key]=dict(mean=statistics.mean(r[key] for r in rows),median=statistics.median(r[key] for r in rows))
    return value


def validate_reports(base,reports,read):
    cases=[];blocks=[];calibrations=[];total=0
    for name,result in reports.items():
        assert result['diagnosticOnly'] is True and result['nativeThread']==result['pid']
        frequency=result['frequency'];prefix=read(base/name/'priming.json')
        prior=0
        for phase in ('before','after'):
            calibration=read(base/name/f'calibration-{phase}.json')
            assert calibration['diagnosticOnly'] and calibration['rusageBytes']==144
            assert calibration['phase']==phase and calibration['pid']==result['pid']
            assert calibration['nativeThread']==result['nativeThread'] and calibration['frequency']==frequency
            assert len(calibration['clocks'])==128
            records=[]
            for i,clock in enumerate(calibration['clocks']):
                assert clock['iteration']==i
                records.append(inspect(clock,frequency,True))
                assert clock['counters']['begin']>prior;prior=clock['counters']['end']
            if phase=='before':assert prior<prefix['began']
            else:assert calibration['clocks'][0]['counters']['begin']>result['rows'][-1]['clocks'][-1]['counters']['end']
            calibrations.append(dict(process=name,phase=phase,**summarize(records)))
        prior=prefix['began']
        groups=[('prefix',p['round'],p['rows']) for p in prefix['passes']]
        groups.append(('suffix',None,result['rows']))
        for phase,round_index,rows in groups:
            for row in rows:
                rank=len(row['shape'])
                payload=4*prod(n+row['pads'][i]+row['pads'][rank+i] for i,n in enumerate(row['shape']))
                values=[]
                assert len(row['clocks'])==780
                for i,clock in enumerate(row['clocks']):
                    assert clock['iteration']==i and clock['warmup']==(i<600)
                    entry=inspect(clock,frequency)
                    assert entry['allocated']>=payload
                    c=clock['counters'];assert c['begin']>prior;prior=c['end']
                    if phase=='prefix':assert c['start']==clock['start'] and c['stop']==clock['stop']
                    values.append(entry)
                total+=len(values)
                for first in range(0,780,60):
                    blocks.append(dict(process=name,phase=phase,round=round_index,case=row['name'],
                        first=first,measured=phase=='suffix' and first>=600,
                        **summarize(values[first:first+60])))
                if phase=='suffix':
                    cohorts=defaultdict(list)
                    for entry in values[600:]:
                        key=f"fault={entry['minor']>0 or entry['major']>0},gc={entry['collection']},switch={entry['voluntary']>0 or entry['involuntary']>0}"
                        cohorts[key].append(entry)
                    cases.append(dict(process=name,case=row['name'],output_payload_bytes=payload,
                        **summarize(values[600:]),cohorts={key:summarize(group) for key,group in sorted(cohorts.items())}))
    assert len(cases)==48 and len(calibrations)==8 and len(blocks)*60==total
    return dict(passed=True,calls=total,calibration_calls=1024,cases=cases,blocks=blocks,calibrations=calibrations)
