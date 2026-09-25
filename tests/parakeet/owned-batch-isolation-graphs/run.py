"""Reuse graph transport with the relocation's preparation and namespace."""
import importlib.util
import sys
from prepare import BASE, PARENT, TOOLS, prepare

loader = importlib.util.spec_from_file_location('graph_transport', PARENT/'run.py')
transport = importlib.util.module_from_spec(loader)
loader.loader.exec_module(transport)
REMOTE = '/dev/shm/lokad-parakeet-owned-batch-isolation-graphs-20260925'
transport.PRELUDE = transport.PRELUDE.replace(transport.REMOTE, REMOTE)
transport.BASE, transport.REMOTE, transport.TOOLS = BASE, REMOTE, TOOLS

if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare','stage','launch','observe','collect']
    getattr(transport, sys.argv[1])()
