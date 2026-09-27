"""Reproduce the closed decoder observation summary without running inference."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-decoder-projection-observation-v3-amd-20260927'
DESTINATION = Path(__file__).with_name('observations-20260927.json')


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def summarize():
    closed = read(BASE / 'closed.json')
    assert pin(BASE / 'closed.json')['sha256'] == 'ef94387e0517b08fd87d010c08d99e0b801e1bd7d2809c04b1dcec614333397d'
    assert closed['passed'] and closed['remote_terminal'] and not closed['performance_admitted']
    for name, identity in closed['files'].items():
        assert pin(BASE / name) == identity, name
    value = read(BASE / 'analysis.json')
    assert value['diagnostic_only'] and not value['product_changed']
    trace = read(BASE / 'collected/trace-capture/result.json')
    state = read(BASE / 'collected/identity.json')
    exported = read(BASE / 'collected/trace-export/events/summary.json')
    categories = ['one-row-row-major', 'prepared-final-row', 'packing', 'other-matmul-kernel', 'other', 'empty']
    aggregates = {}
    for label, low, high in [('warmup', 0, 256), ('observed', 256, 1280), ('all', 0, 1280)]:
        totals = dict.fromkeys(categories, 0.0)
        calls = set()
        for row in value['stack_intervals']['inside']:
            if low <= row['iteration'] < high:
                totals[row['category']] += row['estimated_thread_ms']
                if row['category'] == 'one-row-row-major' and row['estimated_thread_ms'] > 0:
                    calls.add(row['iteration'])
        total = sum(totals.values())
        aggregates[label] = dict(calls=high-low, categories_ms=totals, total_ms=total,
            row_major_fraction=totals['one-row-row-major']/total,
            calls_with_row_major_sample_intervals=len(calls))
    path, = (BASE / 'collected/trace-stacks').glob('*.speedscope.json')
    document = read(path)
    frames = document['shared']['frames']
    worker = next(p for p in document['profiles'] if p['name'] == f"Thread ({trace['native_thread']})")
    stack, example = [], None
    for event in worker['events']:
        if event['type'] == 'O':
            stack.append(event['frame'])
            if 'mm_m1_kblocked' in frames[event['frame']]['name']:
                example = [frames[index]['name'] for index in stack]
                break
        else:
            assert stack.pop() == event['frame']
    audit_files = [ROOT / ('artifacts/parakeet-decoder-projection-observation-v3-' + name)
        for name in ['audit-20260927.stdout', 'audit-20260927.stderr',
            'offline-audit-20260927.py', 'offline-audit-20260927.stdout', 'offline-audit-20260927.stderr',
            'offline-audit-v2-20260927.py', 'offline-audit-v2-20260927.stdout', 'offline-audit-v2-20260927.stderr']]
    return dict(passed=True, diagnostic_only=True, product_changed=False, performance_admitted=False,
        closure=pin(BASE / 'closed.json'), analysis=pin(BASE / 'analysis.json'),
        product=value['product'], consumer=value['consumer'],
        supervisor=state['supervisor'], started=state['started'], ended=state['ended'],
        jobs=[dict(name=r['name'], code=r['code']) for r in state['runs']],
        resource_observations=sum(r['samples'] for r in value['resources']),
        peak_owned_rss=max(r['peak_rss'] for r in value['resources']),
        reports=value['reports'], mapping=value['mapping'], operands=value['operands'],
        exported_records=exported['recorded'], lost_events=exported['lost'],
        markers=value['events']['markers'], method_records=len(value['method_events']),
        sampled_intervals=aggregates, example_row_major_stack=example,
        outside=value['stack_intervals']['outside'],
        rounding_adjustments_ms=value['stack_intervals']['rounding_adjustments_ms'],
        per_call_jit_version_proved=False, per_node_allocation_or_copy_proved=False,
        node_association='Named managed MatMul stack plus the exact original graph and source guards; no per-node marker.',
        counter_scope=trace['counter_scope'],
        sample_scope=value['stack_intervals']['scope'],
        audit_platform='Frozen auditor; unavailable local psutil fails on any access; only remote command paths use PurePosixPath. Linux resource observations are unchanged.',
        audit_invocations={p.relative_to(ROOT).as_posix(): pin(p) for p in audit_files})


if __name__ == '__main__':
    assert sys.argv[1:] in [[], ['--publish']]
    result = summarize()
    if sys.argv[1:]:
        with DESTINATION.open('x', encoding='utf8', newline='\n') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
    else:
        assert read(DESTINATION) == result
    print(json.dumps(dict(passed=True, closure=result['closure'],
        sampled_observation=result['sampled_intervals']['observed'], performance_admitted=False)))
