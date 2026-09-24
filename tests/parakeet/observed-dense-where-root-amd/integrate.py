"""Apply the exact admitted product and its public tests after all release gates."""
import json,shutil
from prepare import ROOT,TOOLS,APPLIED,SOURCE,PRIOR,GRAPH,gates
from protocol import pin,save
from source_scope import verify_before,delta,root_files,TEST


def main():
    assert not APPLIED.exists();source=gates()
    verify_before(source);changed=[*delta(source),TEST];files=root_files(source)
    APPLIED.mkdir()
    for name in changed:
        assert (ROOT/name).resolve().is_relative_to(ROOT)
        if (ROOT/name).exists():
            backup=APPLIED/'before'/name;backup.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,backup)
    intended=dict(prepared=pin(SOURCE/'prepared.json'),prerequisites={label:pin(folder/'closed.json') for label,folder in PRIOR.items()},
        graph_qualification=pin(GRAPH/'closed.json'),
        before={name:source['before'].get(name) for name in changed},source_files=files,changed=changed)
    save(APPLIED/'intended.json',intended)
    for name in changed:
        target=ROOT/name;temporary=target.with_suffix(target.suffix+'.m70tmp');assert not temporary.exists()
        origin=TOOLS/'DenseScalarWhereTests.cs.txt' if name==TEST else SOURCE/'source'/name
        temporary.write_bytes(origin.read_bytes());assert pin(temporary)==files[name];temporary.replace(target)
    for name,wanted in files.items():assert pin(ROOT/name)==wanted,name
    save(APPLIED/'applied.json',dict(passed=True,root_build_pending=True,**intended))
    print(json.dumps(dict(passed=True,changed=changed,source_files=len(files),applied=pin(APPLIED/'applied.json'))))


if __name__=='__main__':main()
