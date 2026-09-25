"""Reuse the existing bounded event supervisor for eight fixed e5 processes."""
import gzip
import hashlib
import shutil
import remote_base as supervisor
from remote_base import BASE, DOTNET, FLAGS, read, pin, save, live, idle
from protocol import ROLES, KEYS
from checks import compiled_scope


def command_for(name, spec):
    role, action = name.split('-'); assert role in ROLES
    if action == 'capture':
        return [DOTNET, BASE/'runtimes/candidate/ReleaseBenchmark.dll', BASE/'cases-candidate.json',
                KEYS[role], BASE/name/'output', 'timing'], False, 2
    assert action == 'export'
    return [DOTNET, BASE/'export-runtime/DispatchEventsExport.dll',
            BASE/(role+'-capture/capture.nettrace'), BASE/name/'events'], False, 0


def after(name, spec, row):
    if name.endswith('-capture'):
        role = name.split('-')[0]; product = ROLES[role]
        value = read(BASE/name/'output/result.json'); diagnostic = read(BASE/name/'output/diagnostic.json')
        ready = read(BASE/name/'ready.json'); enabled = read(BASE/name/'collector-enabled.json')
        assert value['passed'] and value['mode']=='timing' and diagnostic['diagnosticOnly']
        assert value['pid']==diagnostic['pid']==row['processes']['worker']['pid']==ready['pid']==enabled['pid']
        assert diagnostic['nativeThread']==ready['native_thread'] and ready['counter']<enabled['counter']
        assert value['runtime']=='10.0.8' and value['flags']=={} and value['key']==KEYS[role]
        assert value['consumer']==read(BASE/'built.json')['consumer']['sha256']
        assert value['core']==spec['products'][product]['Lokad.Onnx.dll']['sha256']
        assert value['calls']==len(value['clocks'])==len(diagnostic['clocks'])==780
        for i, (clock, observation) in enumerate(zip(value['clocks'],diagnostic['clocks'],strict=True)):
            assert clock['index']==observation['index']==i and clock['warmup']==(i<600)
            assert clock['ticks']==observation['end']-observation['start']>0
        expected = read(BASE/f'evidence/original-{KEYS[role]}-{product}.json')
        assert value['arrays']==expected['arrays'] and value['inputs_unchanged'] and value['held_outputs_unchanged']
        assert (BASE/name/'capture.nettrace').stat().st_size>0
    if name.endswith('-export'):
        value = read(BASE/name/'events/summary.json')
        assert value['complete'] and value['lost']==0 and value['clr_events']>0 and value['protocol']=='all-event-records-v1'
        assert value['input_sha256']==pin(BASE/(name.split('-')[0]+'-capture/capture.nettrace'))['sha256']


supervisor.command_for = command_for
supervisor.after = after
if __name__=='__main__': raise SystemExit(supervisor.main())
