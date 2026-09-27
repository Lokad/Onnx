"""Reuse the original Pyannote transport and all terminal identity checks."""
import importlib.util
import sys
from prepare import BASE, TOOLS, ROOT

PARENT = TOOLS.parent/'pad-current-pyannote-amd'
loader = importlib.util.spec_from_file_location('pyannote_transport', PARENT/'run.py')
transport = importlib.util.module_from_spec(loader); loader.loader.exec_module(transport)
REMOTE = '/dev/shm/lokad-parakeet-decoder-packed-row-pyannote-20260927'
transport.PRELUDE = transport.PRELUDE.replace(transport.REMOTE, REMOTE)
transport.BASE, transport.REMOTE, transport.TOOLS = BASE, REMOTE, TOOLS
prepared = transport.prepared

if __name__ == '__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','stage','launch','observe','collect']
    if sys.argv[1]=='observe': assert not (BASE/'closed.json').exists()
    getattr(transport,sys.argv[1])()
