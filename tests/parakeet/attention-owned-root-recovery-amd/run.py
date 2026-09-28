"""Use the unchanged root transport for the single measured attention preparation integration."""
import importlib.util
import sys
from prepare import BASE, ROOT, TOOLS, PARENT, prepare, previous_closed, monitor

loader = importlib.util.spec_from_file_location('prepared_row_root_transport', PARENT/'run.py')
transport = importlib.util.module_from_spec(loader); loader.loader.exec_module(transport)
REMOTE = '/dev/shm/lokad-attention-owned-root-recovery-20260928'
transport.PRELUDE = transport.PRELUDE.replace(transport.REMOTE, REMOTE)
transport.BASE, transport.TOOLS, transport.REMOTE = BASE, TOOLS, REMOTE
prepared, ssh, PRELUDE = transport.prepared, transport.ssh, transport.PRELUDE


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare','stage','launch','observe','collect']
    getattr(transport, sys.argv[1])()
