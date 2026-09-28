"""Reuse the complete-model transport with one fresh transpose namespace."""
import importlib.util
import sys
from prepare import BASE, PARENT, TOOLS

loader = importlib.util.spec_from_file_location('transpose_model_transport', PARENT/'run.py')
transport = importlib.util.module_from_spec(loader); loader.loader.exec_module(transport)
REMOTE = '/dev/shm/lokad-sigmoid-avx512-models-20260928'
transport.PRELUDE = transport.PRELUDE.replace(transport.REMOTE, REMOTE)
transport.BASE, transport.REMOTE, transport.TOOLS = BASE, REMOTE, TOOLS
prepared = transport.prepared
if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare','stage','launch','observe','collect']
    getattr(transport, sys.argv[1])()
