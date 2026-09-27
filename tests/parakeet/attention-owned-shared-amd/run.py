"""Reuse the complete shared-model transport for the single attention ownership pair."""
import importlib.util
import sys
from prepare import BASE, TOOLS

PARENT = TOOLS.parent/'owned-batch-isolation-shared-amd'
loader = importlib.util.spec_from_file_location('shared_transport', PARENT/'run.py')
transport = importlib.util.module_from_spec(loader)
loader.loader.exec_module(transport)
REMOTE = '/dev/shm/lokad-attention-owned-shared-20260928'
transport.PRELUDE = transport.PRELUDE.replace(transport.REMOTE,REMOTE)
transport.BASE, transport.REMOTE, transport.TOOLS = BASE, REMOTE, TOOLS
prepared = transport.prepared

if __name__ == '__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','stage','launch','observe','collect']
    if sys.argv[1]=='observe': assert not (BASE/'closed.json').exists()
    getattr(transport,sys.argv[1])()
