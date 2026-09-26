"""Reuse retained transport; bind a fresh namespace and prospective resource caps."""
from pathlib import Path
import sys
from prepare import ROOT, TOOLS, BASE, OLD

# Retain the transport exactly except its two staging preflight constants.
# All running-job limits come from this diagnostic's frozen protocol.py.
source = (OLD / 'run.py').read_text(encoding='utf8')
for before, after in [('>=12*1024**3', '>=4*1024**3'), ('>=3*1024**3', '>=1*1024**3')]:
    assert source.count(before) == 1
    source = source.replace(before, after)
scope = dict(__name__='retained_transport', __file__=str(OLD / 'run.py'))
exec(compile(source, str(OLD / 'run.py'), 'exec'), scope)
remote = '/dev/shm/lokad-parakeet-pad-warmup-diagnostic-20260926'
scope['PRELUDE'] = scope['PRELUDE'].replace(scope['REMOTE'], remote)
scope.update(ROOT=ROOT, TOOLS=TOOLS, BASE=BASE, REMOTE=remote)
prepared = scope['prepared']

if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'stage', 'launch', 'observe', 'collect']
    scope[sys.argv[1]]()
