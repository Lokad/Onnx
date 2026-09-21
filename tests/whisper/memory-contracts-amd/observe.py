"""Read existing recorded process births and results without restarting work."""
from common import PRELUDE,ssh


def main():
    print(ssh(PRELUDE+'''state=read(base/'run/identity.json')
def live(b):
 try:return psutil.Process(b['pid']).create_time()==b['birth']
 except psutil.NoSuchProcess:return False
births=[state['supervisor']]+[dict(pid=int(p),birth=b) for p,b in state['members'].items()]
print(json.dumps(dict(complete=state['complete'],code=state['code'],error=state.get('error'),
 births=[dict(**b,live=live(b)) for b in births],samples=state['samples'],peak_rss=state['peak_rss'],
 recordings=len(list((base/'run/worker').glob('[0-9][0-9]-*.json'))),result=(base/'run/worker/result.json').exists(),
 stderr=(base/'supervisor.stderr').read_text()[-1500:])))
'''))


if __name__=='__main__':main()
