def schedule(spec):
    assert len(spec['requests'])==21 and spec['requests'][20]['name']==spec['requests'][0]['name']
    return [dict(engine=engine,request=index,name=item['name'],id=f"{engine}-{index:02}-{item['name']}")
            for engine in ['managed','native'] for index,item in enumerate(spec['requests'])]

def coverage(jobs,spec):
    expected=schedule(spec)
    assert spec['schedule']==expected and jobs==expected and len({j['id'] for j in jobs})==42
