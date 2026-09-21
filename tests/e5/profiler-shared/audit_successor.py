"""Reuse the unchanged complete-model auditor and bind the preflight refusal."""
from common import *
import audit


def main():
    target=ROOT/'artifacts/e5-profiler-shared-v2-20260921'
    spec=read(target/'manifest.json');verify(spec);failed=read(BASE/'processes.json')
    assert spec['preflight_wait_seconds']==3600 and spec['predecessor']['state']==pin(BASE/'processes.json')
    assert spec['predecessor']['manifest']==pin(BASE/'manifest.json')
    assert failed['complete'] and failed['code']==1 and failed['runs']==[] and absent(failed['supervisor'])
    assert spec['predecessor']['identity']==failed['supervisor'] and spec['predecessor']['inference_calls']==0
    waits=[json.loads(line) for line in (target/'preflight-waits.jsonl').read_text().splitlines()]
    assert all(w['seconds']<3600 and w['job'] in [j['id'] for j in JOBS] for w in waits)
    write(target/'predecessor-check.json',dict(passed=True,identity=failed['supervisor'],inference_calls=0,resource_wait_samples=len(waits),state=pin(BASE/'processes.json')))
    audit.BASE=target
    audit.main()


if __name__=='__main__':main()
