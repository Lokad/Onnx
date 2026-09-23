from fractions import Fraction

def process_mean(value):
    assert value['mode']=='timing' and value['calls']==len(value['clocks'])==120
    for index,c in enumerate(value['clocks']):
        assert c['index']==index and c['warmup']==(index<60)
        assert type(c['ticks']) is int and type(c['frequency']) is int and c['ticks']>0 and c['frequency']>0
    return sum((Fraction(c['ticks'],c['frequency']) for c in value['clocks'][60:]),Fraction())/60

def summarize(reports):
    assert list(reports)==['current-a','ort-a','ort-b','current-b']
    means={k:process_mean(v) for k,v in reports.items()}
    controls=[];roles={}
    for role in ['current','ort']:
        values=[means[k] for k in [role+'-a',role+'-b']]
        ratio=max(values)/min(values);roles[role]=sum(values)/2
        controls.append(dict(role=role,ratio=float(ratio),passed=ratio<=Fraction(11,10)))
    return dict(means={k:float(v) for k,v in means.items()},current=float(roles['current']),ort=float(roles['ort']),ratio=float(roles['current']/roles['ort']),controls=controls,qualified=all(c['passed'] for c in controls))
