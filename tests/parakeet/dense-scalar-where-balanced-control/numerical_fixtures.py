"""Prospective numerical census; no worker timing informs these cases."""
import json
from pathlib import Path


def base_cases(capture):
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
    masks=[[0,2],[0,255],[2,0],[2,1],[0,1],[0,0],[1,1],[255,255]]
    for index,mask in enumerate(masks):
        add(f'raw-bool-{index}',[2],[],[2],eligible=len({b!=0 for b in mask})==1)
        result[-1]['raw_mask']=mask
    assert len(result)==131 and len({r['name'] for r in result})==131
    return result


def inherited_cases(capture):
    import math
    result=base_cases(capture)
    def add(name,c,x,y,mask='false',layout='dense',eligible=True,error=None):
        result.append(dict(name=name,dtype='Float',cshape=c,xshape=x,yshape=y,mask=mask,layout=layout,eligible=eligible,error=error))
    for n in [4095,4096,4097]:
        for mask in ['false','true','first','last','alternating']:
            add(f'boundary-{n}-{mask}',[n],[],[n],mask,eligible=mask in ['false','true'])
    for mask in ['false','true']:
        add('large-nonscalar-'+mask,[4096],[4096],[4096],mask,eligible=False)
    masks=[[0,2],[0,255],[2,0],[2,1],[0,1],[0,0],[1,1],[255,255]]
    for i,mask in enumerate(masks):
        add(f'large-raw-bool-{i}',[2,1],[],[2,2048],eligible=len({b!=0 for b in mask})==1)
        result[-1]['raw_mask']=mask
    for layout in ['offset','condition-custom','x-custom','y-custom','condition-reverse','x-reverse','y-reverse',
                   'condition-view','x-view','y-view','condition-broadcast','alias']:
        for mask in ['false','true']:
            add('large-'+layout+'-'+mask,[8192],[1],[8192],mask,layout,eligible=layout in ['offset','alias'])
    for label,c,x,y in [('rank9',[4096],[],[1,1,1,1,1,1,1,1,4096]),
        ('condition-outranks',[1,4096],[],[4096]),('x-outranks',[4096],[1,1,1],[4096])]:
        for mask in ['false','true']:add('large-'+label+'-'+mask,c,x,y,mask,eligible=False)
    for label,c,x,y in [('condition',[2,2049],[],[2,2048]),('x',[1],[4095],[4096]),('both',[5],[4095],[4096])]:
        for mask in ['false','true']:
            add('large-invalid-'+label+'-'+mask,c,x,y,mask,error='System.ArgumentException',eligible=False)
    for seed in range(11):
        add(f'large-scalar-bits-{seed}',[4096],[],[4096],'true');result[-1]['x_seed']=seed
    for c in result:
        c['provider_attempt']=c['dtype']=='Float' and not c['layout'].startswith('null-') and math.prod(c['yshape'])>=4096 and math.prod(c['xshape'])==1
        c['provider_eligible']=c['provider_attempt'] and c['eligible']
    assert len(result)==len({c['name'] for c in result})==203
    assert sum(bool(c.get('error')) for c in result)==15
    assert sum(c['eligible'] for c in result)==73
    assert sum(c['provider_eligible'] for c in result)==35
    return result


def cases(capture):
    import math
    result = inherited_cases(capture)
    for rank in range(1, 9):
        y = [4096] if rank == 1 else [8] + [1 if i % 2 == 0 else 2 for i in range(rank - 2)] + [512]
        shapes = [y.copy(), y[:-1] + [1], [y[-1]], [1 if i % 2 == 0 else d for i, d in enumerate(y)]]
        for label, c, mask in zip(['full', 'last-scalar', 'trailing', 'alternating-axes'], shapes,
                                  ['first', 'last', 'alternating', 'alternating'], strict=True):
            result.append(dict(name=f'run-rank{rank}-{label}', dtype='Float', cshape=c, xshape=[],
                               yshape=y, mask=mask, layout='dense', error=None))
    for c in result:
        c['eligible'] = (c['dtype'] == 'Float' and not c.get('error')
                         and 1 <= len(c['yshape']) <= 8
                         and len(c['cshape']) <= len(c['yshape'])
                         and len(c['xshape']) <= len(c['yshape'])
                         and math.prod(c['xshape']) == 1
                         and c['layout'] in ['dense', 'offset', 'alias'])
        c['provider_attempt'] = (c['dtype'] == 'Float' and not c['layout'].startswith('null-')
                                 and math.prod(c['yshape']) >= 4096 and math.prod(c['xshape']) == 1)
        c['provider_eligible'] = c['provider_attempt'] and c['eligible']
    assert len(result) == len({c['name'] for c in result}) == 235
    assert sum(bool(c.get('error')) for c in result) == 15
    assert sum(c['eligible'] for c in result) == 152
    assert sum(c['provider_attempt'] for c in result) == 123
    assert sum(c['provider_eligible'] for c in result) == 95
    return result
