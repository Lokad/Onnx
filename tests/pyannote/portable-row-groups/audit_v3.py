import common
common.BASE = common.ROOT / 'artifacts/pyannote-portable-row-groups-v3-20260921'
common.monitor.BASE = common.BASE
from audit import main

if __name__ == '__main__':
    st=common.read(common.BASE/'processes.json')
    assert st['complete'] and st['code']==0
    common.terminal(st['supervisor'])
    shapes=common.read(common.BASE/'shapes.json')['shapes']
    spec=common.read(common.BASE/'prepared.json')
    for job in spec['jobs']:
        result=common.read(common.BASE/'output'/(job+'.json'))
        rows=result['conditioning']
        if job=='validate':assert rows==[];continue
        assert [(r['m'],r['n'],r['k']) for r in rows]==[(r['m'],r['n'],r['k']) for r in shapes]
        assert all(r['seconds']>=1 and r['calls']>=16 and r['calls']%16==0 for r in rows)
    main()
