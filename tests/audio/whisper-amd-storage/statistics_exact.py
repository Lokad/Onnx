"""Independent report reconstruction from integer ticks, without stored seconds."""
from decimal import Decimal, localcontext


def exact_rows(workers, cases):
    assert len(workers) == 4 and len({c['name'] for c in cases}) == len(cases) and cases
    assert [w['engine'] for w in workers] == ['managed','ort','ort','managed']
    rows = []
    with localcontext() as context:
        context.prec = 50
        for name, selected in [('complete-corpus',cases)]+[(c['name'],[c]) for c in cases]:
            names = {c['name'] for c in selected}
            duration = Decimal(sum(c['samples'] for c in selected))/16000; assert duration > 0
            row = dict(name=name,audio_seconds=duration)
            for engine in ['managed','ort']:
                visits = []
                for worker in workers:
                    if worker['engine'] != engine:
                        continue
                    totals = []
                    for iteration in [1,2,3]:
                        records = [r for r in worker['records'] if r['pass']==iteration and r['name'] in names]
                        assert len(records) == len(selected) and {r['name'] for r in records} == names
                        assert all(type(r[k]) is int for r in records for k in ['start_ticks','end_ticks','frequency'])
                        assert all(r['frequency'] > 0 and r['end_ticks'] > r['start_ticks'] for r in records)
                        totals.append(sum((Decimal(r['end_ticks']-r['start_ticks'])/Decimal(r['frequency']) for r in records),Decimal(0)))
                    visits.append(dict(passes=totals,mean=sum(totals)/3))
                assert len(visits) == 2
                seconds = sum(v['mean'] for v in visits)/2
                row[engine] = dict(seconds=seconds,rtf=seconds/duration,visits=visits)
            row['ratio'] = row['managed']['seconds']/row['ort']['seconds']; rows.append(row)
    return rows


def suffix(row):
    return ' | '+' | '.join(f'{float(value):.3f}' for value in [row['managed']['seconds'],row['ort']['seconds'],row['ratio'],row['managed']['rtf'],row['ort']['rtf']])+' |'
