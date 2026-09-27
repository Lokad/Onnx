"""Reuse the complete request transport for the fixed rational sigmoid comparison."""
import importlib.util
import sys
from prepare import BASE, TOOLS, TRANSPORT

loader = importlib.util.spec_from_file_location('application_transport', TRANSPORT/'run.py')
transport = importlib.util.module_from_spec(loader)
loader.loader.exec_module(transport)
REMOTE = '/dev/shm/lokad-parakeet-rational-sigmoid-app-20260927'
transport.PRELUDE = transport.PRELUDE.replace(transport.REMOTE, REMOTE)
transport.BASE, transport.REMOTE, transport.TOOLS = BASE, REMOTE, TOOLS
prepared = transport.prepared

if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare','stage','launch','observe','collect']
    getattr(transport, sys.argv[1])()
