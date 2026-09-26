"""Reconcile the retained exact-model ORT Pad allocator counters, without inference."""
from collections import Counter
import csv
from functools import reduce
import hashlib
import json
from math import prod
from operator import xor
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-ort-diagnosis-amd-20260924'
APP = ROOT / 'artifacts/parakeet-prepared-recurrence-app-amd-20260924'
ORT = '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'
INPUTS = {}


def pin(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read(path, wanted=None):
    data = path.read_bytes()
    if wanted is not None:
        assert pin(data) == wanted, path
    INPUTS[path.relative_to(ROOT).as_posix()] = pin(data)
    return json.loads(data)


def main():
    closure = read(BASE / 'closed.json')
    assert INPUTS[(BASE/'closed.json').relative_to(ROOT).as_posix()]['sha256'] == \
        '615353b4d8524f8935076357c0949859553c27826d50d53ebd82fb002d146a85'
    assert closure['passed']
    receipt = read(BASE/'collected/collection.json', closure['collection'])
    assert receipt['terminal'] and receipt['code'] == 0
    observation = read(BASE/'collected/profile/observation.json', receipt['files']['profile/observation.json'])
    profile = observation['profiles']['encoder']
    assert receipt['files']['profile/'+profile['file']] == {k:profile[k] for k in ('bytes','sha256')}
    events = read(BASE/'collected/profile'/profile['file'], receipt['files']['profile/'+profile['file']])
    spec = read(BASE/'spec.json')
    app_closed = read(APP/'closed.json', spec['previous_closure'])
    manifest = read(APP/'collected/manifests/current-parakeet.json', spec['manifest'])
    adapter = APP/'collected/runtime/audio_adapter.py'
    data = adapter.read_bytes()
    assert pin(data) == app_closed['files']['collected/runtime/audio_adapter.py']
    assert pin(data) == {k:manifest['adapter'][k] for k in ('bytes','sha256')}
    INPUTS[adapter.relative_to(ROOT).as_posix()] = pin(data)
    source = {}
    names = ['core/providers/cpu/tensor/pad.cc', 'core/framework/execution_frame.cc',
             'core/framework/session_state.cc', 'core/framework/session_options.h',
             'core/framework/sequential_executor.cc', 'core/framework/bfc_arena.cc',
             'core/framework/resource_accountant.cc']
    for name in names:
        path = 'onnxruntime/'+name
        data = subprocess.run(['git','-c','gc.auto=0','-C',str(ROOT/'external/onnxruntime'),
            'show',ORT+':'+path],capture_output=True,check=True).stdout
        source[path] = pin(data)

    runs = sorted([e for e in events if e['name']=='model_run'], key=lambda e:e['ts'])
    pads = [e for e in events if e.get('args',{}).get('op_name')=='Pad']
    calls = [c for c in observation['calls'] if c['graph']=='encoder']
    assert len(runs)==len(calls)==len(observation['requests'])==80 and len(pads)==3840
    assert all(a['ts']+a['dur'] < b['ts'] for a,b in zip(runs,runs[1:]))
    rows=[]; requests=[]; seen_keys=set(); names=None
    for i,(run,call,request) in enumerate(zip(runs,calls,observation['requests'],strict=True)):
        assert call['request']==request['index']==i and request['iteration']==i//20
        assert call['inputs']==calls[i%20]['inputs']
        key=reduce(xor,(dim for tensor in call['inputs'].values() for dim in tensor['shape']),0)
        subset=sorted([e for e in pads if run['ts']<=e['ts'] and
                       e['ts']+e['dur']<=run['ts']+run['dur']],key=lambda e:e['ts'])
        assert len(subset)==48
        present={e['name'] for e in subset}
        assert len(present)==48 and (names is None or names==present)
        names=present; frame_count=call['outputs']['outputs']['shape'][2]
        for e in subset:
            a=e['args']; assert e['pid']==e['tid']==run['pid']==run['tid']
            assert a['provider']=='CPUExecutionProvider'
            shape=a['output_type_shape'][0]['float']
            family='attention' if '/self_attn/' in e['name'] else 'convolution'
            assert '/depthwise_conv/' in e['name'] or family=='attention'
            expected=([1,8,frame_count,2*frame_count] if family=='attention'
                      else [1,1024,frame_count+8])
            assert shape==expected and int(a['output_size'])==4*prod(shape)
            row=dict(request=i,pass_index=i//20,measured=i>=20,clip=request['name'],
                feed_frames=call['inputs']['audio_signal']['shape'][2],frame_count=frame_count,
                memory_pattern_key=key,prior_key=key in seen_keys,node=e['name'],
                node_index=int(a['node_index']),family=family,ts_us=e['ts'],duration_us=e['dur'],
                output_bytes=int(a['output_size']))
            row.update({k:int(v) for k,v in a.items() if k.startswith('mem_')})
            assert row['mem_arena_held_delta']==0
            assert row['mem_requested_in_use_delta']==(row['output_bytes'] if i<20 else 0)
            assert (row['mem_in_use_delta']==0)==(i>=20)
            rows.append(row)
        requests.append(dict(index=i,clip=request['name'],inputs=call['inputs'],
            pattern_key=key,prior_key=key in seen_keys,pad_calls=48,
            pad_requested_delta=sum(int(e['args']['mem_requested_in_use_delta']) for e in subset),
            pad_time_us=sum(e['dur'] for e in subset)))
        assert (key in seen_keys)==(i>=20)
        seen_keys.add(key)
    assert len(seen_keys)==20 and len(rows)==3840
    groups=[]
    for measured in (False,True):
        for family in ('attention','convolution'):
            subset=[r for r in rows if r['measured']==measured and r['family']==family]
            groups.append(dict(measured=measured,family=family,calls=len(subset),
                zero_requested_delta=sum(r['mem_requested_in_use_delta']==0 for r in subset),
                requested_delta_bytes=sum(r['mem_requested_in_use_delta'] for r in subset),
                output_payload_bytes=sum(r['output_bytes'] for r in subset),
                time_us=sum(r['duration_us'] for r in subset)))
    result=dict(diagnostic_only=True,inference_calls=0,product_changed=False,inputs=INPUTS,
        generator=pin(Path(__file__).read_bytes()),ort_revision=ORT,source=source,
        allocator_route='Warm calls obtain live arena space; measured calls add no per-Pad arena space. '
                        'Memory-pattern explanation is source-derived; addresses and branch PCs were not captured.',
        requests=requests,groups=groups,unique_feed_shapes=20,unique_pattern_keys=20)
    with (OUT/'ort-allocation-observations-20260926.json').open('x',encoding='utf8') as f:
        f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (OUT/'ort-pad-allocations-20260926.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps(dict(groups=groups,rows=len(rows),unique_keys=len(seen_keys))))


if __name__=='__main__':main()
