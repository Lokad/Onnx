"""Check the changed lifecycle and passive memory observations, then all full arrays."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'input-cross'))
import audit as original
from common import read

def metadata(result,spec):
    assert spec['protocol']=='whisper-input-cross-reused-context-v2'
    assert result['context_lifecycle']=='one-reused-context'
    assert len(result['records'])==42
    previous=None
    for row in result['records']:
        before=row['memory_before'];after=row['memory_after']
        for value in [before,after]:
            assert set(value)=={'allocated_total','managed_estimate','collections','gc_index','last_gc_heap','last_gc_fragmented','last_gc_committed'}
            assert all(type(value[k]) is int and value[k]>=0 for k in value if k!='collections')
            assert len(value['collections'])==3 and all(type(v) is int and v>=0 for v in value['collections'])
            assert value['last_gc_fragmented']<=value['last_gc_heap']<=value['last_gc_committed']
            if previous is not None:
                assert value['allocated_total']>=previous['allocated_total'] and value['gc_index']>=previous['gc_index']
                assert all(a>=b for a,b in zip(value['collections'],previous['collections']))
            previous=value
        assert type(row['pool_allocated_bytes']) is int and row['pool_allocated_bytes']>=0
        assert type(row['pool_reused_bytes']) is int and row['pool_reused_bytes']>=0
        assert after['allocated_total']-before['allocated_total']>=row['pool_allocated_bytes']
    return dict(first= result['records'][0]['memory_before'],last=result['records'][-1]['memory_after'],
                pool_allocated_bytes=sum(r['pool_allocated_bytes'] for r in result['records']),
                pool_reused_bytes=sum(r['pool_reused_bytes'] for r in result['records']))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();summary=metadata(read(args.artifact/'managed/result.json'),read(args.artifact/'manifest.json'))
    original.main()
    print('Verified context reuse/allocation metadata:',summary)

if __name__=='__main__':main()
