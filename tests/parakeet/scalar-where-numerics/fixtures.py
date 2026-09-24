"""Prospective numerical census; no worker timing informs these cases."""
import json
from pathlib import Path


def cases(capture):
    result = []
    for index, f in enumerate(capture['fixtures']):
        shapes = [i['tensor']['shape'] for i in f['inputs']]
        dtype = f['inputs'][1]['tensor']['dtype']
        result.append(dict(name=f'capture-{index}', dtype=dtype, cshape=shapes[0], xshape=shapes[1], yshape=shapes[2],
            mask='captured', layout='dense', files=[i['file'] for i in f['inputs']], reference=f['output']['file'], eligible=dtype=='Float'))
        if dtype == 'Float':
            for mask in ['true', 'first', 'last', 'alternating']:
                result.append(dict(name=f'capture-{index}-{mask}', dtype=dtype, cshape=shapes[0], xshape=shapes[1], yshape=shapes[2],
                    mask=mask, layout='dense', files=[i['file'] for i in f['inputs']], eligible=mask=='true'))
    def add(name, c, x, y, mask='false', layout='dense', dtype='Float', error=None, eligible=True):
        result.append(dict(name=name, dtype=dtype, cshape=c, xshape=x, yshape=y, mask=mask,
                           layout=layout, error=error, eligible=eligible))
    for mask in ['false','true','first','last','alternating']:
        for label,c,x,y in [('vector',[32],[],[32]),('scalar-mask',[],[1],[2,4,32]),
            ('outer-mask',[2,1,1],[],[2,4,32]),('inner-broadcast',[2,1,32],[1,1],[2,4,32]),
            ('rank8',[1,1,1,1,1,1,1,3],[],[2,1,2,1,1,1,1,3])]:
            n=1
            for d in c:n*=d
            add(f'{label}-{mask}',c,x,y,mask,eligible=mask in ['false','true'] or n==1)
    for label,c,x,y in [('scalar-result',[],[],[]),('rank9',[3],[],[1,1,1,1,1,1,1,1,3]),
        ('condition-outranks',[2,3],[],[3]),('x-outranks',[3],[1,1,1],[2,3]),
        ('nonscalar-x',[2,1],[2,3],[3])]:
        for mask in ['false','true']:
            add(f'{label}-{mask}',c,x,y,mask,eligible=False)
    for index,(c,x,y) in enumerate([([0],[],[0]),([1],[],[2,0,3]),([1,0,1],[1,1],[2,0,3]),([0,3],[1],[2,0,3])]):
        add(f'empty-{index}',c,x,y)
    for layout in ['offset','condition-custom','x-custom','y-custom','condition-reverse','x-reverse','y-reverse',
                   'condition-view','x-view','y-view','condition-broadcast','alias']:
        for mask in ['false','true']:
            c,y=([32],[32]) if layout.endswith(('-custom','-reverse')) else ([1,1,32],[2,4,32])
            add(f'{layout}-{mask}',c,[1],y,mask,layout,eligible=layout in ['offset','alias'])
    for dtype in ['Bool','Byte','Int32','Int64','UInt32','UInt64','Double','Half']:
        add(f'dtype-{dtype}',[2,1],[1],[2,3],'alternating',dtype=dtype,eligible=False)
    for mask in ['false','true']:
        for label,c,x,y in [('condition',[2,5],[],[2,3]),('x',[1],[2],[3]),('both',[5],[2],[3])]:
            add(f'invalid-{label}-{mask}',c,x,y,mask,error='System.ArgumentException',eligible=False)
    for operand in ['condition','x','y']:
        add(f'null-{operand}',[3],[],[3],layout='null-'+operand,error='System.NullReferenceException',eligible=False)
    for seed in range(11):
        add(f'scalar-bits-{seed}',[32],[],[32],'true')
        result[-1]['x_seed']=seed
    assert len(result)==123 and len({r['name'] for r in result})==123
    return result
