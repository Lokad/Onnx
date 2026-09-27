"""Reuse the established single-process trace transport in a fresh namespace."""
import importlib.util
import sys
from prepare import TOOLS

source = TOOLS.parent/'decoder-projection-observation/run.py'
loader = importlib.util.spec_from_file_location('pointwise_runtime_transport', source)
transport = importlib.util.module_from_spec(loader); loader.loader.exec_module(transport)
REMOTE = '/dev/shm/lokad-pwt-runtime-20260927'
assert len(REMOTE+'/tmp/dotnet-diagnostic-'+'9'*10+'-'+'9'*20+'-socket') < 108
transport.PRELUDE = transport.PRELUDE.replace(transport.REMOTE, REMOTE)
transport.REMOTE = REMOTE
transport.transport.PRELUDE, transport.transport.REMOTE = transport.PRELUDE, REMOTE
prepared, ssh, PRELUDE = transport.prepared, transport.ssh, transport.PRELUDE
if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'stage', 'launch', 'observe', 'collect']
    getattr(transport, sys.argv[1])()
