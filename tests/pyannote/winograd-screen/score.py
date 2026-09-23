from fractions import Fraction
from gates import ORDER,evaluate,encode

COMPONENT='937ab50e8d9cc140d7c27ba9d010bd8fe32638bc115173e436ef44a9646edd84'

def fixtures(spec):
    calls=[c for c in spec['calls'] if c['eligible'] and c['attributes']['strides']==[1,1]]
    assert len(calls)==87 and len({(c['case'],c['index']) for c in calls})==87
    assert sorted({c['form'] for c in calls})==[1,2,5,6,9,10,13,14]
    return calls

def clock(row):
    assert type(row['ticks']) is type(row['frequency']) is int
    assert row['ticks']>0 and row['frequency']>0
    return Fraction(row['ticks'],row['frequency'])

def check_call(row,call,expected,role,pass_,iteration):
    n,c,h,w=call['input']['shape'];m=call['weights']['shape'][0]
    assert n==1 and call['output']['shape']==[1,m,h,w]
    work=m*c*9*h*w;iterations=max(1,((1<<31)+work-1)//work);assert iterations==3
    assert row['kind']=='call' and row['role']==role and row['pass']==pass_ and row['warmup']==(pass_==0)
    assert (row['fixture'],row['index'],row['form'])==(call['case'],call['index'],call['form'])
    assert row['iteration']==iteration and row['iterations']==iterations and row['work']==work
    assert row['values']==m*h*w and row['exact'] and row['sha256']==expected['output' if role=='candidate' else 'selectedOutput']
    requested=4*(16*c*8+16*m*8+m*h*w) if role=='candidate' else 4*(c*(h+2)*(w+2)+m*h*w)
    assert row['scratch_requested_bytes']==requested<=64*1024**2
    assert requested<=row['scratch_rented_bytes']<=2*requested
    return clock(row)/(3*iterations)

def constants(calls):
    result={}
    for c in calls:result.setdefault(c['index'],c)
    assert len(result)==29
    return list(result.values())

def check_preparation(rows,calls,role,passes):
    unique=constants(calls);assert len(rows)==passes*29
    hashes={};total=Fraction();size=0
    for index,row in enumerate(rows):
        pass_=index//29;call=unique[index%29];m,c,kh,kw=call['weights']['shape']
        expected=4*m*c*(16 if role=='candidate' else 9)
        assert row['kind']=='preparation' and row['role']==role and row['pass']==pass_ and row['warmup']==(pass_==0)
        assert row['index']==call['index'] and row['bytes']==expected
        assert len(row['sha256'])==64 and hashes.get(row['index'],row['sha256'])==row['sha256']
        hashes[row['index']]=row['sha256'];duration=clock(row)
        if pass_==0:size+=expected
        else:total+=duration/(passes-1)
    assert size<=64*1024**2
    return size,encode(total)

def check_result(value,calls,reference):
    role=value['role'];assert role in ['current','candidate','verify']
    assert value['passed'] and value['read_only_operands'] and value['held_outputs']
    assert value['protocol']=='winograd-87-geometry-2pow31-v1' and value['component']==COMPONENT
    assert value['cases']==87
    expected={(r['fixture'],r['index']):r for r in reference['rows']};assert len(expected)==87
    if role=='verify':
        assert value['calls']==value['warmups']==174 and value['measured']==0
        assert len(value['observations'])==174 and len(value['preparation'])==58
        for j,leg in enumerate(['current','candidate']):
            size,_=check_preparation(value['preparation'][j*29:(j+1)*29],calls,leg,1)
            for row,call in zip(value['observations'][j*87:(j+1)*87],calls,strict=True):
                check_call(row,call,expected[(call['case'],call['index'])],leg,0,0)
                assert row['prepared_bytes']==size
        return dict(passed=True,calls=174)
    assert value['calls']==1044 and value['warmups']==261 and value['measured']==783 and len(value['observations'])==1044
    size,prep=check_preparation(value['preparation'],calls,role,4)
    totals={c['form']:Fraction() for c in calls};index=0
    for pass_ in range(4):
        for call in calls:
            for iteration in range(3):
                row=value['observations'][index];index+=1
                t=check_call(row,call,expected[(call['case'],call['index'])],role,pass_,iteration)
                assert row['prepared_bytes']==size
                if pass_:totals[call['form']]+=t
    return totals,prep,size

def score(reports,spec,reference):
    assert list(reports)==ORDER
    calls=fixtures(spec);totals={};prep={};sizes={}
    for name,value in reports.items():
        assert value['role']==name.split('-')[0]
        totals[name],prep[name],sizes[name]=check_result(value,calls,reference)
    value=evaluate(totals,{c['form']:True for c in calls})
    assert len(value['controls'])==18 and len(value['gates'])==9
    value.update(preparation=prep,prepared_bytes=sizes,calls=4176,warmups=1044,measured=3132,preparations=464)
    return value
