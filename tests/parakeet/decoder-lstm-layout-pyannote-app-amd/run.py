"""Use the original serial transport in the fixed LSTM layout namespace."""
import importlib.util
import sys
from prepare import BASE, TOOLS, PARENT, prepare, previous_closed, monitor

loader = importlib.util.spec_from_file_location('pyannote_application_transport', PARENT/'run.py')
transport = importlib.util.module_from_spec(loader); loader.loader.exec_module(transport)
REMOTE = '/dev/shm/lokad-lstmlayout-pyannote-app-20260927'
transport.PRELUDE = transport.PRELUDE.replace(transport.REMOTE, REMOTE)
transport.BASE, transport.TOOLS, transport.REMOTE = BASE, TOOLS, REMOTE
prepared, ssh, PRELUDE = transport.prepared, transport.ssh, transport.PRELUDE


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare','stage','launch','observe','collect']
    getattr(transport, sys.argv[1])()
