"""Observe an existing e5 phase by actual process birth; never start a phase."""
import argparse
from remote import PRELUDE, ssh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['aa', 'compare'], required=True)
    args = parser.parse_args()
    script = PRELUDE+'''
phase=%r
deployment=read(base/('deployment-'+phase+'.json'))
def live(identity):
 try:return psutil.Process(identity['pid']).create_time()==identity['birth']
 except psutil.NoSuchProcess:return False
path=base/('result-'+phase)/'identity.json'
if not path.exists():
 print(json.dumps(dict(initializing=True,supervisor_live=live(deployment),deployment=deployment,stderr=(base/('supervisor-'+phase+'.stderr')).read_text()[-2000:])));sys.exit()
state=read(path);latest=state['runs'][-1] if state['runs'] else None
if latest is not None:
 current=path.parent/latest['job']['name']/'identity.json'
 if current.exists():latest=read(current)
births=[] if latest is None else [dict(pid=int(p),birth=b,live=live(dict(pid=int(p),birth=b))) for p,b in latest['members'].items()]
print(json.dumps(dict(complete=state['complete'],code=state.get('code'),error=state.get('error'),supervisor_live=live(deployment),
 started_workers=len(state['runs']),completed_workers=sum(r.get('code')==0 for r in state['runs']),
 latest=None if latest is None else dict(job=latest['job'],samples=latest['samples'],peak_rss=latest['peak_rss'],code=latest.get('code'),births=births),
 stderr=(base/('supervisor-'+phase+'.stderr')).read_text()[-2000:])))
''' % args.phase
    print(ssh(script))


if __name__ == '__main__':
    main()
