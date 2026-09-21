"""Check all22shape membership and all conditioning before the unchanged timing audit."""
from common import *

if __name__=='__main__':
    state=read(BASE/'processes.json');assert state['complete'] and state['code']==0;terminal(state['supervisor'])
    shapes=read(BASE/'shapes.json')['shapes']
    coverage=read(ROOT/'artifacts/pyannote-convolution-epilogue-20260921/analysis.json')
    assert len(shapes)==22 and {tuple(r[k] for k in ['m','n','k']) for r in shapes}=={tuple(r[k] for k in ['m','n','k']) for r in coverage['all_tile_shapes']}
    spec=read(BASE/'prepared.json')
    for job in spec['jobs']:
        result=read(BASE/'output'/(job+'.json'));rows=result['conditioning']
        if job=='validate':assert rows==[];continue
        assert [(r['m'],r['n'],r['k']) for r in rows]==[(r['m'],r['n'],r['k']) for r in shapes]
        assert all(r['seconds']>=1 and r['calls']>=16 and r['calls']%16==0 for r in rows)
    path=ROOT/'tests/pyannote/portable-row-groups/audit.py'
    namespace=dict(__name__='complete_shape_timing_auditor',__file__=str(path))
    exec(compile(path.read_text(encoding='utf8'),str(path),'exec'),namespace)
    namespace['main']()
