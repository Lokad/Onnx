"""Preserve the preflight refusal and validate the completed control exactly."""
import json
from checks import PROFILE, partial, pin
from resume import write


def main():
    assert not (PROFILE/'closed.json').exists()
    value=partial()
    proof=dict(passed=False,preserved_failure=True,control_valid=True,refusal=value['refusal'],
        control_requests=len(value['results']['control']['records']),resources=value['resources'],
        original_transfer=pin(PROFILE/'capture-transfer.json'),auditor=pin(__file__),
        files={p.relative_to(PROFILE).as_posix():pin(p) for p in PROFILE.rglob('*') if p.is_file()})
    write(PROFILE/'closed.json',proof)
    print(json.dumps(dict(closure=pin(PROFILE/'closed.json'),refusal=value['refusal'],resources=value['resources'])))


if __name__=='__main__': main()
