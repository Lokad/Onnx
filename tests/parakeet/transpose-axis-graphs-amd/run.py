"""Use the unchanged transport in the corrected, separately frozen namespace."""
import importlib.util
import sys
from prepare import BASE,TOOLS,PARENT,prepare,previous_closed

loader=importlib.util.spec_from_file_location('pad_graph_transport',PARENT/'run.py')
transport=importlib.util.module_from_spec(loader);loader.loader.exec_module(transport)
REMOTE='/dev/shm/lokad-transpose-axis-graphs-20260928'
transport.PRELUDE=transport.PRELUDE.replace(transport.REMOTE,REMOTE)
transport.BASE,transport.REMOTE,transport.TOOLS=BASE,REMOTE,TOOLS
prepared,ssh,PRELUDE=transport.prepared,transport.ssh,transport.PRELUDE


if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','stage','launch','observe','collect']
    getattr(transport,sys.argv[1])()
