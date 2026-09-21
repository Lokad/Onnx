"""Finalize controller metadata only after every writer has exited."""
import json
from common import *


def main():
    finish=read(BASE/'finish-state.json');assert finish['complete'] and finish['code']==0;terminal(finish['supervisor'])
    assert [s['phase'] for s in finish['stages']]==['run','audit']
    for stage in finish['stages']:
        assert stage['complete'] and stage['code']==0;terminal(stage['worker'])
    state=read(BASE/'processes.json');assert state['complete'] and state['code']==0;terminal(state['supervisor'])
    identities=[finish['supervisor'],state['supervisor']]+[s['worker'] for s in finish['stages']]
    for row in state['runs']:
        assert row['complete'] and row['code']==0 and row['application_passed']
        for pid,birth in row['members'].items():
            identity=dict(pid=int(pid),birth=birth);terminal(identity);identities.append(identity)
    closure=read(BASE/'closed.json');assert closure['passed'];files=dict(closure['files'])
    active={p.relative_to(ROOT).as_posix() for p in (BASE/'finish-state.json',BASE/'audit-supervisor.log')}
    finalized={}
    for name,wanted in files.items():
        actual=pin(ROOT/name)
        if name in active:finalized[name]=dict(while_writer_active=wanted,final=actual)
        else:assert actual==wanted,name
    assert set(finalized)==active
    for name,row in finalized.items():files[name]=row['final']
    prepared=read(BASE/'prepared.json')
    for name,wanted in prepared['external_files'].items():assert pin(Path(name))==wanted,name
    for p in (BASE/'closed.json',Path(__file__)):files[p.relative_to(ROOT).as_posix()]=pin(p)
    analysis=read(BASE/'analysis.json');assert analysis['calls']==640
    assert not (BASE/'completion.json').exists()
    save(BASE/'completion.json',dict(passed=True,attribution_valid=analysis['attribution_valid'],selected_for_amd=analysis['selected_for_amd'],
        files=files,terminal_identities=identities,finalized_controller_metadata=finalized,
        scope='All inference, audit and finish writers terminal; active metadata finalized without changing any timing or correctness evidence'))
    print(json.dumps(dict(passed=True,attribution_valid=analysis['attribution_valid'],selected_for_amd=analysis['selected_for_amd'],completion=pin(BASE/'completion.json'))))


if __name__=='__main__':main()
