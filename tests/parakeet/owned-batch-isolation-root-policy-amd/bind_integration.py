"""Record current root bytes and the original admissions after the policy correction."""
from source_scope import ROOT,CORRECTION,FAILED,SOURCE,verify_source,root_files
from protocol import pin,read,save


def main():
    out=CORRECTION/'integration';assert not out.exists()
    source=verify_source();files=root_files(source)
    for name,wanted in files.items():assert pin(ROOT/name)==wanted,name
    prior=read(FAILED/'bundle/evidence/root-applied.json')
    changed=sorted(name for name,wanted in files.items() if source['before'].get(name)!=wanted)
    assert len(changed)==16 and set(files)==set(prior['source_files'])
    value=dict(prior,source_files=files,changed=changed,prepared=pin(SOURCE/'prepared.json'),
        policy_correction=pin(CORRECTION/'applied.json'),failed_root=pin(FAILED/'closed.json'),
        root_build_pending=True)
    out.mkdir();save(out/'applied.json',value)
    print(dict(passed=True,source_files=len(files),changed=len(changed),receipt=pin(out/'applied.json')))


if __name__=='__main__':main()
