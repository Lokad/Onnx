"""Publish limits of the closed sigmoid observation without another inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-sigmoid-residual-diagnostic-amd-20260928'
OUT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def main():
    target = OUT/'observations-20260928.json'
    assert not target.exists()
    assert pin(BASE/'closed.json')['sha256'] == '31b91fba1f513c219a7f1c4b9c44b637ff90389d676522856b64be83c5280a54'
    closure = read(BASE/'closed.json')
    assert closure['passed'] and closure['analysis'] == pin(BASE/'analysis.json')
    for name, wanted in closure['files'].items():
        assert pin(BASE/name) == wanted, name
    repair = read(BASE/'audit-reader-repair.json')
    assert repair['adapter'] == pin(ROOT/'tests/parakeet/sigmoid-residual-diagnostic-amd/audit_gzip.py')
    value = read(BASE/'analysis.json')
    stacks = value['sigmoid_samples']
    public = next(r['seconds']/3 for r in stacks['inclusive'] if '.Sigmoid(' in r['method'])
    helper = next(r['seconds']/3 for r in stacks['inclusive'] if '.SigmoidRationalVector(' in r['method'])
    public_leaves = [dict(r, seconds_per_corpus=r['seconds']/3) for r in stacks['exclusive'] if '.Sigmoid(' in r['method']]
    public_self = sum(r['seconds_per_corpus'] for r in public_leaves)
    other_children = public-helper-public_self
    assert other_children >= 0
    by_reason = Counter(); counts = Counter()
    for pause in value['events']['suspensions']:
        counts[pause['reason']] += 1
        for interval in value['events']['request_intervals']:
            overlap = max(0, min(pause['end_ms'], interval['end_ms'])-max(pause['begin_ms'], interval['begin_ms']))
            by_reason[pause['reason']] += overlap/3000
    assert abs(sum(by_reason.values())-value['events']['request_suspension_seconds_per_corpus']) < 1e-8
    loops = []; listings = {}
    for role in ['control', 'sampled']:
        path = BASE/'collected/logs'/f'{role}-target.log'
        listings[role] = pin(path)
        text = path.read_text(encoding='utf8')
        for part in text.split('; Assembly listing for method '):
            if not part.startswith('Lokad.Onnx.CPUExecutionProvider:SigmoidRationalVector'):
                continue
            if not part.splitlines()[0].endswith('(Tier1)'):
                continue
            body = part[part.index('G_M000_IG03:'):part.index('G_M000_IG04:')]
            instructions = re.findall(r'^\s+([0-9A-F]+)\s+([a-z][a-z0-9]+)\s*(.*?)$', body, re.M)
            counts_op = Counter(op for raw,op,args in instructions)
            assert len(instructions) == 50 and sum(len(raw)//2 for raw,op,args in instructions) == 316
            assert counts_op['vfmadd213ps'] == 9 and counts_op['vdivps'] == 1 and counts_op['vfixupimmps'] == 5
            assert 'zmm' not in body and 'ymm' in body and not counts_op['call']
            loops.append(dict(role=role, tier='Tier1', bits=256, elements_per_iteration=8,
                bytes=316, instructions=50, opcodes=dict(counts_op), calls_in_vector_body=0))
    assert len(loops) == 2
    artifact = ROOT/'tests/parakeet/transpose-axis-profile-results/activation-breakdown-20260928.json'
    native = read(artifact)['retained_native_dispatch']
    result = dict(passed=True, inference_calls=0, product_changed=False, conclusion='Cause remains unresolved; no product candidate selected.',
        inputs={p.relative_to(ROOT).as_posix():pin(p) for p in [BASE/'closed.json', BASE/'analysis.json', BASE/'audit-failed.json', BASE/'audit-reader-repair.json', BASE/'review-initial-assertion.json', artifact, Path(__file__)]},
        timings=value['timings'], sampled_to_control=value['sampled_to_control'],
        control_to_previous_uninstrumented=value['control_to_previous_uninstrumented'],
        sampled_to_wall=value['sampled_to_wall'], events=value['events']['events'], lost=value['events']['lost'],
        request_markers=value['events']['markers'], requests=value['requests'],
        sampled_sigmoid_seconds_per_corpus=dict(public_inclusive=public, rational_helper=helper,
            public_self=public_self, other_children=other_children,
            outside_helper=public-helper, helper_fraction_of_public=helper/public),
        public_leaves=public_leaves, suspension_seconds_per_corpus_by_reason=dict(by_reason),
        paired_suspension_counts=dict(counts), unmatched_suspension_events=value['events']['unpaired_suspension_events'],
        current_helper_loops=loops, current_listings=listings, native_execution=native,
        current_sigmoid_method_events=len(value['events']['sigmoid_method_events']),
        limits=['Method stacks do not resolve instructions, active tier or native allocation work.',
            'CPU_TIME is a stack-export label, not independent CPU measurement.',
            'SuspendOther must not be relabeled as garbage collection; preserve unmatched events.',
            'Do not subtract profiler overhead or combine independent node and stack clocks.',
            'A width difference is proved; its recoverable application cost is not.'],
        next='Inspect retained raw stack-address and method/native mapping evidence before another inference. Resolve public Sigmoid self samples; no vector-width, fusion or allocation experiment yet.')
    with target.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(passed=True, report=pin(target), seconds=result['sampled_sigmoid_seconds_per_corpus'], pauses=dict(by_reason))))


if __name__ == '__main__':
    main()
