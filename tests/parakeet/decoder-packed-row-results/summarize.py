"""Reproduce the fixed candidate's compiled and arithmetic contract summary."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-decoder-packed-row-contracts-v3-amd-20260927'
FIRST = ROOT/'artifacts/parakeet-decoder-packed-row-contracts-amd-20260927'
SECOND = ROOT/'artifacts/parakeet-decoder-packed-row-contracts-v2-amd-20260927'
DESTINATION = Path(__file__).with_name('contracts-20260927.json')


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def summarize():
    prerequisites = {}
    for folder, name, digest in [
        (BASE, 'closed.json', 'fc00688c8ef0e65e5d5109f1808209add023e740aabc5b4312647e547df7d61d'),
        (FIRST, 'failed.json', '6225233c00528cde3170a535ac973ddc7382c3af5086b370d90ff860d023d24d'),
        (SECOND, 'failed.json', '2c74d4aede20993b14a988c8a04399f238954f0c6455c78395e1cb1cda581478')]:
        assert pin(folder/name)['sha256'] == digest
        value = read(folder/name)
        for file, wanted in value['files'].items(): assert pin(folder/file) == wanted, file
        prerequisites[folder.name] = pin(folder/name)
    value = read(BASE/'analysis.json')
    assert value['passed'] and not value['root_product_changed'] and not value['performance_admitted']
    rows = []
    for mode, roles in value['contracts'].items():
        for role, report in roles.items():
            captured = next(r for r in report['public_cases'] if r['name'] == 'captured-projection')
            rows.append(dict(mode=mode, role=role, public_cases=len(report['public_cases']),
                public_output_values=sum(r['checked_values'] for r in report['public_cases']),
                raw_cases=len(report['raw_cases']), raw_output_values=sum(r['checked_values'] for r in report['raw_cases']),
                every_output_exact=all(r['exact'] for r in report['public_cases']+report['raw_cases']),
                kernel_allocated_bytes=sum(r['allocated_bytes_for_eight_calls'] for r in report['raw_cases']),
                captured_projection=captured))
    jobs = []
    for folder in [FIRST, SECOND, BASE]:
        state = read(folder/'collected/identity.json')
        assert state['complete']
        jobs.extend(dict(campaign=folder.name, name=r['name'], code=r['code'], resource_observations=r['samples'],
            peak_observed_owned_rss=r['peak_rss']) for r in state['runs'])
    return dict(passed=True, root_product_changed=False, performance_measured=False,
        products=value['products'], consumer=value['consumer'], compiled=value['compiled'],
        public_cases=sum(r['public_cases'] for r in rows), raw_cases=sum(r['raw_cases'] for r in rows),
        public_output_values=sum(r['public_output_values'] for r in rows), raw_output_values=sum(r['raw_output_values'] for r in rows),
        modes=rows, jobs=jobs, prerequisites=prerequisites,
        limitations='Operator arithmetic and memory contracts only. Public allocation figures are the recorded correctness calls, not steady-state averages. Full-model and application qualification and performance measurements remain pending.')


if __name__ == '__main__':
    assert sys.argv[1:] in [[], ['--publish']]
    result = summarize()
    if sys.argv[1:]:
        with DESTINATION.open('x', encoding='utf8', newline='\n') as stream:
            json.dump(result, stream, indent=2); stream.write('\n')
    else:
        assert read(DESTINATION) == result
    print(json.dumps({k: result[k] for k in ['passed', 'products', 'public_cases', 'raw_cases', 'public_output_values', 'raw_output_values', 'performance_measured']}))
