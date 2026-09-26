"""Read closed sigmoid evidence; publish observations without another experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-sigmoid-execution-diagnostic-amd-20260927'
ORDER = ['current-0', 'candidate-1', 'candidate-2', 'current-3']
METHOD = re.compile(r'^; Assembly listing for method ([^\n]+) \(([^\n]+)\)\n(.*?); Total bytes of code (\d+)[^\n]*', re.M | re.S)
INSTRUCTION = re.compile(r'^\s+([0-9A-F]{2,30})\s+(.+)$', re.M)


def pin(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def instructions(block):
    return [m[2].strip() for m in INSTRUCTION.finditer(block)]


def summarize():
    closure = read(BASE / 'closed.json')
    assert pin(BASE / 'closed.json')['sha256'] == 'e2ec74acf9b8fe9c36a48226aa478b5efded38071f6b699a04f64410265dd868'
    assert pin(BASE / 'analysis.json') == closure['analysis']
    a = read(BASE / 'analysis.json')
    assert a['passed'] and a['diagnostic_only'] and not a['admitted']
    assert a['samples'] == 143520 and a['no_clock_trimmed']
    folder = BASE / 'capture-collected'
    assert pin(folder / 'capture-collection.json') == closure['collection']
    collection = read(folder / 'capture-collection.json')
    assert list(a['observations']) == ORDER
    cases = []
    for index in range(46):
        rows = {name: a['observations'][name]['cases'][index] for name in ORDER}
        name = rows[ORDER[0]]['name']
        assert all(r['name'] == name for r in rows.values())
        cases.append(dict(name=name, elements=rows[ORDER[0]]['elements'],
            equal_minimum_allocation=len({r['minimum_allocated_bytes'] for r in rows.values()}) == 1,
            processes={n: dict(seconds=a['instrumented_case_seconds'][n][index], **r) for n, r in rows.items()}))
    code = {}
    for name in ORDER:
        path = folder / 'logs' / (name + '.stdout')
        assert pin(path) == collection['files']['logs/' + name + '.stdout']
        matches = list(METHOD.finditer(path.read_text(encoding='utf8')))
        assert len(matches) == len(a['codegen'][name]['listings'])
        for m, listing in zip(matches, a['codegen'][name]['listings'], strict=True):
            assert hashlib.sha256(m[0].encode()).hexdigest() == listing['listing_sha256']
        selected = next(m for m in matches if m[2] == 'Tier1' and 'CPUExecutionProvider:Sigmoid(' in m[1])
        blocks = re.split(r'(?=^G_M\d+_IG\d+:)', selected[3], flags=re.M)
        vector = []
        scalar = []
        for i, block in enumerate(blocks):
            if 'vcvttps2dq' in block:
                # Body, store/termination and backedge form the emitted vector loop.
                loop = ''.join(blocks[i:i+3])
                ins = instructions(loop)
                vector.append(dict(assembly=loop, instructions=len(ins),
                    calls=[s for s in ins if s.startswith('call ')],
                    stack_operands=[s for s in ins if re.search(r'\[(?:r|e)(?:sp|bp)[+\-]', s)],
                    ymm_instructions=sum(bool(re.search(r'\bymm\d+\b', s)) for s in ins),
                    zmm_instructions=sum(bool(re.search(r'\bzmm\d+\b', s)) for s in ins)))
            if 'call     System.MathF:Exp(float):float' in block:
                scalar.append(block)
        osr = next(m for m in matches if m[2] == 'Tier1-OSR'
            and 'call     System.MathF:Exp(float):float' in m[3] and 'vcvttps2dq' not in m[3])
        osr_loop = next(b for b in re.split(r'(?=^G_M\d+_IG\d+:)', osr[3], flags=re.M)
            if 'call     System.MathF:Exp(float):float' in b)
        code[name] = dict(raw_stdout=pin(path), vector_width=a['observations'][name]['vector_width'],
            emitted_versions=[dict(tier=m[2], signature=m[1], native_bytes=int(m[4])) for m in matches],
            optimized_sigmoid_bytes=int(selected[4]), vector_loops=vector,
            scalar_exp_blocks=scalar, first_scalar_osr_instructions=instructions(osr_loop))
    assert all(c['equal_minimum_allocation'] for c in cases)
    assert all(v['vector_width'] == 8 for v in code.values())
    for name in ORDER[1:3]:
        loop, = code[name]['vector_loops']
        assert not loop['calls'] and loop['ymm_instructions'] > 0 and not loop['zmm_instructions']
        assert len(loop['stack_operands']) == 1 and 'bword ptr' in loop['stack_operands'][0]
    equal_osr = all(code[n]['first_scalar_osr_instructions'] == code[ORDER[0]]['first_scalar_osr_instructions'] for n in ORDER)
    assert equal_osr
    references = {
        'ort': ROOT / 'artifacts/parakeet-ort-activation-review-20260926/closed.json',
        'profile': ROOT / 'artifacts/parakeet-pad-current-gap-20260927/closed.json',
        'original_rejected_screen': ROOT / 'artifacts/parakeet-vector-sigmoid-screen-amd-20260925/closed.json',
    }
    expected = {'ort': '41f851c4e473856383c6d51f7b5a0e86d5f0ff8c47947df1f4176b35e2301fc3',
        'profile': 'bcb8fff63e202147e96995931e02e0f55f7255ea299cb6b51a3878ec2526d549',
        'original_rejected_screen': 'aaca2b2fd3b54f66dec841b948c0673279b833f90733770348fcbe41dfc4201f'}
    for n, p in references.items():
        assert pin(p)['sha256'] == expected[n]
    return dict(diagnostic_only=True, no_application_score=True, source_closure=pin(BASE / 'closed.json'),
        source_analysis=closure['analysis'], source_build_review=pin(BASE / 'build-review.json'),
        references={n: dict(path=p.relative_to(ROOT).as_posix(), **pin(p)) for n, p in references.items()},
        products=a['products'], consumer=a['consumer'], samples=a['samples'],
        warmup_samples=a['warmup_samples'], measured_samples=a['measured_samples'],
        resources=a['resources'], cases=cases, code=code,
        first_scalar_osr_loop_instructions_equal_ignoring_relative_call_bytes=equal_osr,
        limits=['No instruction sampling or tier-to-clock join.',
            'Equal allocation bytes do not establish equal allocation latency.',
            'Sparse extra allocations are retained; their source is unidentified.',
            'Synthetic fixture values are not captured model intermediates.',
            'The original rejected screen retains its failed verdict.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    result = summarize()
    destination = Path(__file__).with_name('observations-20260927.json')
    text = json.dumps(result, indent=2) + '\n'
    if args.publish:
        with destination.open('x', encoding='utf8', newline='\n') as stream:
            stream.write(text)
    elif destination.exists():
        assert destination.read_text(encoding='utf8') == text
    print(json.dumps(dict(cases=len(result['cases']), equal_allocation_minima=True,
        first_scalar_osr_loop_equal=result['first_scalar_osr_loop_instructions_equal_ignoring_relative_call_bytes'],
        vector_loops={n: [{k: v for k, v in b.items() if k != 'assembly'} for b in c['vector_loops']]
            for n, c in result['code'].items()}, diagnostic_only=True)))
