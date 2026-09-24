"""Preserve the original refusal, then independently qualify its unchanged compiled artifacts."""
import json
from pathlib import Path
import subprocess
import sys
from renames import normalize

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-prepared-recurrence-build-review-20260924'
sys.path.insert(0,str(ROOT/'tests/parakeet/prepared-recurrence-build-amd'))
from run import BASE as BUILD, prepared
from protocol import JOBS, LIMITS, check_sample, pin, read, save
from checks import inventory


def main():
    assert not BASE.exists() and not (BUILD/'closed.json').exists();spec=prepared()
    collected=BUILD/'collected';receipt=read(collected/'collection.json');transfer=read(BUILD/'collection-transfer.json');payload=read(BUILD/'payload.json')
    assert transfer['passed'] and transfer['archive']==pin(BUILD/'results.tar.gz') and transfer['receipt']==pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code']==1 and receipt['input_error'] is None and receipt['payload']==pin(BUILD/'payload.json')
    for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
    for name,wanted in read(BUILD/'bundle/stage.json')['files'].items():assert payload['files'][name]==wanted==pin(BUILD/'bundle'/name),name
    for name,wanted in payload['files'].items():
        if name.startswith(('measured/','bridge/')):assert pin(collected/name)==wanted,name
    state=read(collected/'identity.json')
    assert state['complete'] and state['code']==1 and state['supervisor']==read(BUILD/'deployment.json')
    assert [r['name'] for r in state['runs']]==payload['jobs']==JOBS and state['boot_time']==1789634288.0
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    assert state['error']==state['runs'][-1]['error']
    assert 'checks.py' in state['error'] and "row['public_surface_equal']" in state['error'] and state['error'].endswith('AssertionError\n')
    resources=[]
    for row in state['runs']:
        assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds']
        assert row['preflight']==row['preflight_observations'][-1]
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            # This retained build monitor identifies a job by its unique log
            # path; unlike the later capture monitor it has no row['job'] field.
            check_sample(sample)
            assert all(row['members'][str(m['pid'])]==m['birth'] for m in sample['members'])
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[row['seconds']-samples[-1]['seconds']]
        assert all(0<=g<10 for g in gaps)
        resources.append(dict(name=row['name'],samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    assert state['ended']-state['started']<4*3600
    assert (collected/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    built=read(collected/'built.json');assert built['passed']
    for name,wanted in built['files'].items():
        if name.startswith('runtime/'):assert pin(collected/name)==wanted,name
    for name,wanted in built['product'].items():
        assert pin(collected/'runtime'/name)==wanted==built['files']['source/src/Lokad.Onnx.CLI/bin/Release/net10.0/'+name]
    raw=read(collected/'inventory/instructions.json');normalized,renames=normalize(raw)
    report=inventory(normalized,payload['measured'],built['product'],read(collected/'evidence/prior-composition.json'))
    assert len(renames['renamed_methods'])==28 and len(renames['name_only_methods'])==35
    assert report['candidate_core_methods']==3250 and report['unchanged_core_methods']==3179
    assert len(report['existing_method_changes'])==10 and len(report['added_methods'])==61
    checks=subprocess.run([sys.executable,'-X','utf8','-B',str(TOOLS/'test_renames.py')],text=True,encoding='utf8',capture_output=True)
    assert checks.returncode==0,(checks.stdout,checks.stderr)
    BASE.mkdir()
    (BASE/'tests.stdout').write_text(checks.stdout,encoding='utf8');(BASE/'tests.stderr').write_text(checks.stderr,encoding='utf8')
    # Close the initial refusal once all original evidence is reconciled. Keep
    # its code/reason intact; the successful successor is a separate artifact.
    save(BUILD/'closed.json',dict(passed=False,reason='Scope wrapper did not reconcile compiler identifiers changed by the added graph field.',
        build_jobs_passed=True,inference_calls=0,performance_calls=0,files={p.relative_to(BUILD).as_posix():pin(p) for p in BUILD.rglob('*') if p.is_file()},
        local_inputs=spec['files'],remote_terminal=receipt['identities']))
    save(BASE/'renames.json',renames)
    analysis=dict(passed=True,measured=payload['measured'],built=built['product'],inventory=report,resources=resources,
        source_prepared=pin(collected/'evidence/source-prepared.json'),original_refusal=pin(BUILD/'closed.json'),
        raw_inventory=pin(collected/'inventory/instructions.json'),renames=pin(BASE/'renames.json'),
        renamed_methods=28,name_only_methods=35,rebuilds=0,root_product_changed=False,numerically_qualified=False,no_performance_measurement=True)
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(ROOT).as_posix():pin(p) for folder in [BASE,TOOLS] for p in folder.rglob('*') if p.is_file()}
    files[(BUILD/'closed.json').relative_to(ROOT).as_posix()]=pin(BUILD/'closed.json')
    save(BASE/'closed.json',dict(passed=True,files=files,paths_relative_to_repository=True,original_refusal=pin(BUILD/'closed.json'),
        analysis=pin(BASE/'analysis.json'),remote_terminal=receipt['identities']))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),original_refusal=pin(BUILD/'closed.json'),built=built['product'],
        changed_methods=10,added_methods=61,unchanged_core=3179,unchanged_data=697,compiler_renames=28,rebuilds=0,
        resource_samples=sum(r['samples'] for r in resources),peak_rss=max(r['peak_rss'] for r in resources))))


if __name__=='__main__':main()
