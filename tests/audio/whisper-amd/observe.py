"""Inspect the existing campaign and actual process creation times; never launch."""
from common import PRELUDE,ssh

def main():
    print(ssh(PRELUDE+'''
deployment=read(base/'deployment.json')
def live(b):
 try:return psutil.Process(b['pid']).create_time()==b['birth']
 except psutil.NoSuchProcess:return False
if not (base/'campaign/identity.json').exists():
 print(json.dumps(dict(initializing=True,deployment=deployment,live=live(deployment),stderr=(base/'supervisor.stderr').read_text()[-1500:])));sys.exit()
state=read(base/'campaign/identity.json');runs=[]
for r in state['runs']:
 folder=base/r['output']
 runs.append(dict(name=r['name'],complete=r['complete'],code=r['code'],error=r.get('error'),samples=r['samples'],peak_rss=r['peak_rss'],
  calls=len(list((folder/'worker').glob('[0-9][0-9][0-9].json'))),
  births=[dict(pid=int(p),birth=b,live=live(dict(pid=int(p),birth=b))) for p,b in r['members'].items()],stderr=(folder/'stderr.txt').read_text()[-1000:]))
print(json.dumps(dict(complete=state['complete'],code=state['code'],error=state.get('error'),supervisor_live=live(state['supervisor']),runs=runs,stderr=(base/'supervisor.stderr').read_text()[-1500:])))
'''))

if __name__=='__main__':main()
