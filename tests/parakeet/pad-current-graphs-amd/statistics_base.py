from fractions import Fraction

def process_mean(value):
    assert value['mode']=='timing' and value['calls']==len(value['clocks'])==780
    for index,c in enumerate(value['clocks']):
        assert c['index']==index and c['warmup']==(index<600)
        assert type(c['ticks']) is int and type(c['frequency']) is int and c['ticks']>0 and c['frequency']>0
    return sum((Fraction(c['ticks'],c['frequency']) for c in value['clocks'][600:]),Fraction())/180

def summarize(reports):
    assert list(reports)==['current-a','candidate-a','ort-a','ort-b','candidate-b','current-b']
    means={k:process_mean(v) for k,v in reports.items()}
    controls=[];roles={}
    for role in ['current','candidate','ort']:
        values=[means[k] for k in [role+'-a',role+'-b']]
        ratio=max(values)/min(values);roles[role]=sum(values)/2
        controls.append(dict(role=role,ratio=float(ratio),passed=ratio<=Fraction(11,10)))
    regression=roles['candidate']/roles['current']
    return dict(means={k:float(v) for k,v in means.items()},current=float(roles['current']),candidate=float(roles['candidate']),
        ort=float(roles['ort']),ratio=float(roles['candidate']/roles['ort']),current_over_ort=float(roles['current']/roles['ort']),
        candidate_over_current=float(regression),regression_passed=regression<=Fraction(105,100),controls=controls,
        qualified=all(c['passed'] for c in controls) and regression<=Fraction(105,100))
