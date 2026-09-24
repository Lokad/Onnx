"""Preserve the selected standalone graph declaration failure; no candidate ran."""
import json
from run import BASE, prepared
from protocol import LIMITS, check_sample, pin, read, save


def main():
    spec=prepared();assert not (BASE/'closed.json').exists()
    collected=BASE/'collected';receipt=read(collected/'collection.json');transfer=read(BASE/'collection-transfer.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code']==1 and receipt['input_error'] is None and receipt['payload']==pin(BASE/'payload.json')
    for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
    state=read(collected/'identity.json')
    assert state['complete'] and state['code']==1 and state['supervisor']==read(BASE/'deployment.json') and state['boot_time']==1789634288.0
    assert [r['name'] for r in state['runs']]==['sdk-version','calls-restore','calls-build','selected-512']
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources=[]
    for row,code in zip(state['runs'],[0,0,0,-6],strict=True):
        assert row['complete'] and row['code']==code and row['seconds']<LIMITS['seconds']
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            assert sample['job']==row['name'];check_sample(sample)
            assert all(row['members'][str(m['pid'])]==m['birth'] for m in sample['members'])
        resources.append(dict(name=row['name'],samples=len(samples),seconds=row['seconds'],peak_rss=row['peak_rss']))
    built=read(collected/'built.json');assert built['passed']
    for name,wanted in built['files'].items():assert pin(collected/name)==wanted,name
    error=(collected/'logs/selected-512.stderr').read_text()
    assert 'Cannot use user input :float:1x1x640 for required input /decoder/Transpose_output_0' in error
    expected=['english-16k','french-44k-stereo','jfk-48k-stereo','english-token-limit','english-frame-limit','english-repeat']
    assert (collected/'logs/selected-512.stdout').read_text().splitlines()==[n+' original decoder passed' for n in expected]
    assert not (collected/'selected-512/output/result.json').exists() and not (collected/'candidate-512').exists()
    result=dict(passed=False,phase='selected standalone LSTM graph input binding',candidate_executions=0,no_performance_measurement=True,
        reason='Standalone inputs were null without descriptors; CheckOneBoundInput correctly requires declared tensor metadata.',
        repair='Declare float input/output tensors with the exact captured shapes in a fresh namespace. Products, fixtures and gates stay unchanged.',
        completed_selected_decoder_cases=expected,consumer=built['consumer'],resources=resources)
    save(BASE/'analysis.json',result)
    save(BASE/'closed.json',dict(**result,files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},
        local_inputs=spec['files'],remote_terminal=receipt['identities'],generator=pin(__file__)))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**result)))


if __name__=='__main__':main()
