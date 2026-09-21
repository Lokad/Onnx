"""Observe the existing local replay by its recorded PID and creation time."""
import json,sys
from prepare import ROOT,BASE,read
sys.path.insert(0,str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil


def live(birth):
    try:return psutil.Process(birth['pid']).create_time()==birth['birth']
    except psutil.NoSuchProcess:return False


folder=BASE/'local';state=read(folder/'identity.json')
value={k:state.get(k) for k in ['complete','code','error','supervisor','child','samples','peak_rss']}
value['supervisor_live']=live(state['supervisor'])
if state.get('child'):value['child_live']=live(state['child'])
value['recording_rows']=len(list((folder/'worker').glob('[0-9][0-9]-*.json')))
value['result_exists']=(folder/'worker/result.json').exists()
value['stderr_tail']=(folder/'stderr.txt').read_text()[-2000:]
value['stdout_tail']=(folder/'stdout.txt').read_text()[-1200:]
print(json.dumps(value))
