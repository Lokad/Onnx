"""Apply exactly the admitted source after verifying all complete regression gates."""
import json,shutil
from prepare import ROOT,APPLIED,SOURCE,PRIOR,gates,graph_qualification
from protocol import pin,read,save
from source_scope import verify_before,delta

def main():
    assert not APPLIED.exists();source=gates()
    verify_before(source);changed=delta(source)
    APPLIED.mkdir()
    for name in changed:
        assert (ROOT/name).resolve().is_relative_to(ROOT)
        if (ROOT/name).exists():
            backup=APPLIED/'before'/name;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,backup)
    intended=dict(prepared=pin(SOURCE/'prepared.json'),prerequisites={label:pin(folder/'closed.json') for label,folder in PRIOR.items()},
        graph_qualification=pin(graph_qualification.BASE/'closed.json'),
        before={name:source['before'].get(name) for name in changed},source_files=source['source'],changed=changed)
    save(APPLIED/'intended.json',intended)
    for name in changed:
        target=ROOT/name;temporary=target.with_suffix(target.suffix+'.m66tmp');assert not temporary.exists()
        temporary.write_bytes((SOURCE/'source'/name).read_bytes());assert pin(temporary)==source['source'][name];temporary.replace(target)
    for name,wanted in source['source'].items():assert pin(ROOT/name)==wanted,name
    save(APPLIED/'applied.json',dict(passed=True,root_build_pending=True,**intended))
    print(json.dumps(dict(passed=True,changed=changed,source_files=425,applied=pin(APPLIED/'applied.json'))))

if __name__=='__main__':main()
