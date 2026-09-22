"""Correct the encoding spelling before builds; preserve the original attempt."""
import sys
import common

common.BASE = common.ROOT / 'artifacts/pyannote-convolution-reduction-v2-20260922'
common.REMOTE = '/dev/shm/lokad-pyannote-convolution-reduction-v2-20260922'
common.monitor.BASE = common.BASE


def main():
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','stage','launch','observe','collect','audit']
    mode=sys.argv[1]
    path=common.TOOLS / (mode+'.py' if mode in ['prepare','audit'] else 'transport.py')
    source=path.read_text(encoding='utf8')
    if mode in ['prepare','audit']:
        assert source.count("encoding='utf8-sig'")==1
        source=source.replace("encoding='utf8-sig'","encoding='utf-8-sig'")
    namespace=dict(__name__='encoding_corrected_successor',__file__=str(path))
    exec(compile(source,str(path),'exec'),namespace)
    namespace['main' if mode in ['prepare','audit'] else mode]()


if __name__=='__main__': main()
