"""Recheck and close the AMD campaign on the host that executed it."""
from pathlib import Path
import argparse
import importlib.util
import time
from common import pin,read,sha,write_new


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True)
    args=parser.parse_args();base=args.artifact.resolve();root=Path(__file__).resolve().parents[3]
    assert not (base/'closed.json').exists()
    spec=importlib.util.spec_from_file_location('frozen_final_auditor',base/'runtime-source/audit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    value=read(base/'audit.json');assert value['profile']=='amd' and value==module.audit(base)
    checks=read(base/'record-checks.json')
    assert checks['passed'] and len(checks['refusals'])==40 and checks['audit_sha256']==sha(base/'audit.json')
    assert checks['checker_sha256']==sha(base/'source/check_results.py')
    frozen=read(base/'frozen.json');staging=read(base/'staging.json')
    assert frozen['source_commit']==staging['source_commit']
    for name,wanted in staging['files'].items():assert pin(base/name)==wanted,name
    record=dict(schema=1,closed=True,execution_passed=True,application_passed=value['application_passed'],profile='amd',
        closed_at=time.time(),source_commit=frozen['source_commit'],all_owned_processes_terminal=True,
        terminal_processes=value['terminal_processes'],audit_sha256=sha(base/'audit.json'),
        failed_attempt_receipt=pin(base/'prior-failures/windows-status.json'),
        windows_memory_failure_receipt=pin(base/'prior-failures/windows-memory.json'),
        files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()},
        sources={p.relative_to(root).as_posix():pin(p) for p in sorted((base/'source').iterdir()) if p.is_file()})
    write_new(base/'closed.json',record)
    print('Closed AMD evidence:',len(record['files']),'files; receipt',sha(base/'closed.json'))


if __name__=='__main__':main()
