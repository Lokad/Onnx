from common import *
from release import released

if __name__=='__main__':
    released()
    path=ROOT/'tests/pyannote/portable-row-groups/run.py'
    namespace=dict(__name__='complete_shape_runner',__file__=str(path))
    exec(compile(path.read_text(encoding='utf8'),str(path),'exec'),namespace)
    namespace['main']()
