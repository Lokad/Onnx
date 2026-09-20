"""Select each declared reference with its own frozen host and runtime identity."""
import ast
import hashlib
import json
from common import pin,read


def profile(base,engine,family):
    if engine in ('native','ort') and family=='whisper' and (base/'native-linux-collected').exists():
        selected=base/'native-linux-collected'
        return selected,read(selected/'manifest.json'),read(selected/'frozen.json')
    return base,read(base/'manifest.json'),read(base/'frozen.json')


def directory(base,engine,family):
    selected,_,_=profile(base,engine,family)
    if selected!=base:return selected/f'process-{engine}-{family}-run'
    if engine=='native' and family=='whisper' and (base/'recovery-plan.json').exists():
        plan=read(base/'recovery-plan.json')
        assert plan['engine']==engine and plan['family']==family
        assert plan['destination']=='process-native-whisper-recovery1'
        assert plan['frozen']==pin(base/'frozen.json')
        return base/plan['destination']
    return base/f'process-{engine}-{family}-run'


def failed_attempt(base):
    if not (base/'recovery-plan.json').exists():return None
    plan=read(base/'recovery-plan.json');frozen=read(base/'frozen.json');limits=frozen['limits']['native']
    assert plan['limits']==limits and plan['frozen']==pin(base/'frozen.json')
    assert plan['cases']==[c['name'] for c in read(base/'manifest.json')['cases']]
    for name,wanted in dict(plan['sources'],**plan['failed_files']).items():assert pin(base/name)==wanted,name
    old=base/'process-native-whisper-run';state=read(old/'identity.json')
    assert state['complete'] is False and state['family']=='whisper' and state['engine']=='native'
    assert state['limits']==limits and state['frozen_sha256']==plan['frozen']['sha256']
    assert 'Available memory limit' in state['error'] and read(old/'complete.json')==dict(code=2)
    samples=[json.loads(s) for s in (old/'samples.jsonl').read_text().splitlines()]
    assert len(samples)==state['samples'] and len(samples)>1
    assert all(s['available']>=limits['available'] for s in samples[:-1]) and samples[-1]['available']<limits['available']
    assert all(0<=s['seconds']<limits['seconds'] for s in samples)
    assert all(a['seconds']<b['seconds'] for a,b in zip(samples,samples[1:]))
    assert max(sum(m['rss'] for m in s['members']) for s in samples)==state['peak_rss']<limits['rss']
    for sample in samples:
        for member in sample['members']:
            assert member['affinity']==[2] and member['birth']==state['members'][str(member['pid'])]
    assert {p.name for p in (old/'worker').iterdir()}=={'00.json'}
    row=read(old/'worker/00.json');assert row['name']=='ES2004a' and row['result']['stop_reason']=='Completed'
    return dict(preserved=True,reason='available memory below fixed guard',completed_cases=[row['name']],
        resource_samples=len(samples),seconds_before_stop=samples[-1]['seconds'],peak_rss=state['peak_rss'],
        last_available=samples[-1]['available'],state=state,completed_record=row,plan=pin(base/'recovery-plan.json'))


def recovery_ready(base):
    if not (base/'recovery-plan.json').exists():return
    plan=read(base/'recovery-plan.json');outcome=read(base/'recovery-outcome.json')
    if (base/'native-linux-collected').exists():
        assert outcome==dict(code=2,child_created=False,reason='Stable preflight not reached',supervisor=read(base/'recovery-supervisor.json'))
        assert not (base/plan['destination']).exists()
        samples=[json.loads(s) for s in (base/'recovery-preflight.jsonl').read_text().splitlines()]
        assert samples and all(a['elapsed']<b['elapsed'] for a,b in zip(samples,samples[1:]))
        assert all(s['stable_seconds']<plan['stable_preflight_seconds']==60 for s in samples)
        assert plan['preflight_timeout_seconds']==900 and 898<=samples[-1]['elapsed']<900
        stable=None
        for sample in samples:
            if sample['available']<plan['limits']['preflight']:stable=None
            elif stable is None:stable=sample['elapsed']
            expected=0 if stable is None else sample['elapsed']-stable
            assert abs(sample['stable_seconds']-expected)<1e-6
        return linux_ready(base)
    assert outcome['code']==0 and outcome['child_created'] is True
    state=read(directory(base,'native','whisper')/'identity.json')
    assert state['supervisor']==outcome['supervisor']==read(base/'recovery-supervisor.json')
    assert plan['created']<state['started'] and plan['limits']==state['limits']
    samples=[json.loads(s) for s in (base/'recovery-preflight.jsonl').read_text().splitlines()]
    assert samples and all(a['elapsed']<b['elapsed'] for a,b in zip(samples,samples[1:]))
    last=samples[-1];assert last['stable_seconds']>=plan['stable_preflight_seconds']==60
    assert last['elapsed']<plan['preflight_timeout_seconds']==900
    first=last['elapsed']-last['stable_seconds']
    assert all(s['available']>=plan['limits']['preflight'] for s in samples if s['elapsed']>=first)


def linux_ready(base):
    selected,manifest,frozen=profile(base,'native','whisper');assert selected!=base
    old=read(base/'manifest.json');launch=read(base/'native-linux-local/launch-check.json')
    assert pin(selected/'frozen.json')==launch['frozen'] and frozen==launch['frozen_value']
    assert manifest==launch['manifest']
    assert manifest['original_frozen']==pin(base/'frozen.json')
    assert {k:manifest['original_manifest'][k] for k in ['bytes','sha256']}==pin(base/'manifest.json')
    for key in ['cases','versions','core_sha256','data_sha256','labels','input_audit','comparison','timing_scope']:
        assert manifest[key]==old[key],key
    assert manifest['families']=={'whisper':old['families']['whisper']} and manifest['schedule']==['whisper']
    assert manifest['limits']==frozen['limits']=={'native':old['limits']['managed']}
    assert manifest['hosts']=={'native':'Linux AMD EPYC 9V74 CPU2'}
    for name,wanted in frozen['files'].items():assert pin(selected/name)==wanted,name
    for name,wanted in read(selected/'predecessor.json')['files'].items():assert pin(base/name)==wanted,name
    dependency=read(selected/'dependency-check.json')
    assert dependency['passed'] is True and pin(selected/'dependency-check.json')==manifest['dependency_check']
    assert manifest['native_files']==frozen['native_files']==dependency['native_files']
    assert manifest['portable_function_body_sha256']==dependency['portable_function_body_sha256']=='43272a768eadeac0fd6eba53bee1c79ec6cf5f86aab3db8af8b6e7cab34893d2'
    assert manifest['borrowed_function_body_sha256']==dependency['borrowed_function_body_sha256']=='e8548ecf6b3c4ee12d748d37e458eafc18628f6e040b4f3c54dd145bacb60f99'
    for name in ['common.py','whisper_recording.py','campaign_processes.py']:
        assert pin(selected/'runtime'/name)==pin(base/'runtime'/name)
    for folder in ['reference-source','upstream']:
        expected={p.relative_to(base/folder).as_posix():pin(p) for p in (base/folder).rglob('*') if p.is_file()}
        actual={p.relative_to(selected/folder).as_posix():pin(p) for p in (selected/folder).rglob('*') if p.is_file()}
        assert actual==expected
    def canonical(node):
        if isinstance(node,ast.AST):return {'type':type(node).__name__,'fields':{k:canonical(v) for k,v in ast.iter_fields(node)}}
        if isinstance(node,list):return [canonical(v) for v in node]
        return node
    def suffix(path):
        main=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='main')
        first=next(i for i,n in enumerate(main.body) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple) and [v.id for v in n.targets[0].elts]==['root','base'])
        return canonical(ast.Module(body=main.body[first:],type_ignores=[]))
    assert suffix(selected/'runtime/native.py')==suffix(base/'runtime/native.py')
    assert hashlib.sha256(json.dumps(suffix(selected/'runtime/native.py'),sort_keys=True).encode()).hexdigest()==manifest['portable_execution_suffix_sha256']
    generator=selected/'reference-source/tests/whisper/recording/generate_reference.py'
    main=next(n for n in ast.parse(generator.read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='main')
    nodes=[n for n in main.body if isinstance(n,ast.FunctionDef) and n.name in ['decode_window','segments_for']]
    assert len(nodes)==2 and pin(generator)==dependency['generator']
    assert hashlib.sha256(json.dumps(canonical(ast.Module(body=nodes,type_ignores=[])),sort_keys=True).encode()).hexdigest()==manifest['portable_function_body_sha256']
    collection=read(selected/'collection.json');check=read(base/'native-linux-collection-check.json')
    assert check['passed'] is True and collection==check['remote']['collection']
    assert pin(selected/'collection.json')==check['remote']['receipt']
    assert pin(base/'native-linux-local/linux-results.tar.gz')==check['remote']['archive']
    assert collection['all_owned_processes_terminal'] is True and collection['frozen']==pin(selected/'frozen.json')
    assert collection['verified_native_files']==frozen['native_files'] and collection['excluded_directory']=='python'
    assert {p.relative_to(selected).as_posix() for p in selected.rglob('*') if p.is_file()}==set(collection['files'])|{'collection.json'}
    for name,wanted in collection['files'].items():assert pin(selected/name)==wanted,name
    assert read(selected/'campaign-native.json')==dict(complete=True,outcomes=[dict(family='whisper',code=0)])
    inputs=selected/'process-native-whisper-inputs'
    assert pin(inputs/'identity.json')==frozen['input_smoke']['state'] and pin(inputs/'worker/inputs.json')==frozen['input_smoke']['inputs']
    assert read(inputs/'worker/inputs.json')==dict(passed=True,family='whisper',affinity=4,cases=[dict(name=c['name'],samples=c['samples'],pcm_sha256=c['pcm_sha256']) for c in manifest['cases']])
    state=read(directory(base,'native','whisper')/'identity.json')
    assert state['supervisor']==launch['state']['supervisor'] and state['child']==launch['state']['child']
    return dict(host=manifest['hosts']['native'],frozen=pin(selected/'frozen.json'),collection=pin(selected/'collection.json'),
        windows_retry=read(base/'recovery-outcome.json'),terminal_processes=collection['terminal_processes'])
