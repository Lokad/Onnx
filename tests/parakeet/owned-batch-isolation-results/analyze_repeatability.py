"""Describe the closed short-e5 failure without rescoring or running inference."""
import csv
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-owned-batch-isolation-graphs-amd-20260925'
ORDER = ['current-a', 'candidate-a', 'ort-a', 'ort-b', 'candidate-b', 'current-b']
CLOSURE = 'def19d3f178cbb318bc11999b6e23dd18db40772d78949707155cf1f4c791638'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    names = ['repeatability-20260925.md', 'repeatability-20260925.json', 'repeatability-blocks-20260925.csv']
    assert all(not (OUT/name).exists() for name in names)
    assert pin(BASE/'closed.json')['sha256'] == CLOSURE
    closed = read(BASE/'closed.json')
    assert closed['passed'] and not closed['admitted'] and not closed['all_controls_passed']
    sources = {}

    def verified(name):
        path = BASE/name
        assert pin(path) == closed['files'][name], name
        sources[name] = pin(path)
        return read(path)

    analysis = verified('analysis.json')
    payload = verified('payload.json')
    state = verified('collected/identity.json')
    assert state['complete'] and state['code'] == 0 and len(state['runs']) == 72
    assert all(r['complete'] and r['code'] == 0 for r in state['runs'])
    failed = [dict(key=row['key'], **control) for row in analysis['performance']
              for control in row['controls'] if not control['passed']]
    assert [(r['key'], r['role']) for r in failed] == [('e5-8tok', 'candidate')]
    assert all(row['regression_passed'] for row in analysis['performance'])
    case, = [row for row in analysis['performance'] if row['key'] == 'e5-8tok']
    blocks, processes = [], []
    for role in ORDER:
        job = 'timing-e5-8tok-'+role
        result = verified('collected/'+job+'/output/result.json')
        run, = [row for row in state['runs'] if row['name'] == job]
        clocks = result['clocks']
        assert len(clocks) == result['calls'] == 780
        assert not result['flags'] and result['inputs_unchanged'] and result['held_outputs_unchanged']
        if not role.startswith('ort'):
            assert result['core'] == payload['products'][role.split('-')[0]]['Lokad.Onnx.dll']['sha256']
        values = []
        for index, clock in enumerate(clocks):
            assert clock['index'] == index and clock['warmup'] == (index < 600)
            value = Fraction(clock['ticks'], clock['frequency'])
            assert value > 0
            values.append(value)
        measured = sum(values[600:])/180
        assert abs(float(measured)-case['means'][role]) < 1e-15
        processes.append(dict(role=role, pid=run['child']['pid'], birth=run['child']['birth'],
                              warmup_seconds=float(sum(values[:600])), measured_seconds=float(sum(values[600:])),
                              measured_mean_seconds=float(measured), worker_seconds=run['seconds']))
        assert result['setup_seconds']+float(sum(values)) <= run['seconds']
        for start in range(0, 780, 30):
            mean = sum(values[start:start+30])/30
            blocks.append(dict(role=role, start=start, end=start+29, count=30,
                               phase='warmup' if start < 600 else 'measured',
                               mean_ms=float(mean*1000), numerator=mean.numerator, denominator=mean.denominator))
    assert len(blocks) == 156 and sum(row['count'] for row in blocks) == 4680
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(blocks[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(blocks)
    evidence = dict(closure=pin(BASE/'closed.json'), sources=sources, failed_controls=failed,
                    original_case=case, processes=processes, blocks=blocks,
                    diagnostic_only=True, inference_run=False, rescored=False, admitted=False)
    lines = [
        '# Dispatch relocation: short-e5 repeatability failure', '',
        'The complete graph campaign is **not admitted**. All numerical checks and',
        'all eight regression gates pass, but only 23 of 24 repeatability controls pass.',
        'The candidate\'s two e5-8tok means are **10.549127 and 12.038724 ms**, ratio',
        '**1.141206**, above the unchanged 1.10 limit. Its 0.997092 mean ratio to',
        'release is therefore not a qualified improvement or proof the old failure is fixed.', '',
        'This analysis partitions every retained short-e5 call into consecutive blocks',
        'of 30. All 4,680 calls appear; no block, warmup or measurement is selected out.',
        'The tables are diagnostic descriptions, not replacement scores.', '',
        '| Calls, zero-based | Phase | Release A ms | Candidate A ms | ORT A ms | ORT B ms | Candidate B ms | Release B ms |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for start in range(0, 780, 30):
        group = {row['role']: row for row in blocks if row['start'] == start}
        lines.append('| '+f"{start}–{start+29} | {group[ORDER[0]]['phase']} | "+
                     ' | '.join(f"{group[role]['mean_ms']:.6f}" for role in ORDER)+' |')
    lines.extend([
        '', 'The difference spans several measured blocks: Candidate A is mostly near',
        '10.1 ms, Candidate B near 11.9 ms, while both ORT processes remain near 6.0 ms.',
        'Release and Candidate A also have slower blocks. These clocks do not identify',
        'a particular compilation, collection, CPU-frequency or native-code event.', '',
        'The earlier [exact-product tier observation](../../benchmarks/e5-direct-tier-results/report-20260925.md)',
        'already showed release f95a13c5 and parent 40260aef receiving final matrix',
        'tiers 13.910–15.148 seconds after the first call, beyond the original prefix.',
        'Those are different, instrumented processes and the candidate there is not',
        'e07a4518. They motivate checking the short benchmark\'s runtime phase; they',
        'cannot assign a tier or historical cause to the present uninstrumented calls.', '',
        'Restoring the shared dispatcher\'s compiled instructions did not establish',
        'repeatable short-e5 performance. Stop this relocation\'s release path; the',
        'conditional Parakeet model stage has not been prepared or executed. Preserve',
        'this verdict and the parent M78 failure. No unchanged scored retry, nearby',
        'product variant, threshold change or retroactive warmup change follows.', '',
        'The next question is whether the scored short-call prefix measures a stable',
        'runtime phase for the exact current products. Inspect retained compilation',
        'evidence before proposing a bounded observation or prospective measurement',
        'correction. Any such correction requires its own evidence and cannot relabel',
        'this failed campaign as passed. No new optimization is selected.', '',
        '[Full verdict and all clocks](graphs-20260925.md),',
        '[exact block fractions](repeatability-blocks-20260925.csv),',
        '[bound evidence](repeatability-20260925.json).', '',
        'Closure: `'+CLOSURE+'`.',
    ])
    documents = [('\n'.join(lines)+'\n'), json.dumps(evidence, indent=2, allow_nan=False)+'\n', output.getvalue()]
    for name, contents in zip(names, documents, strict=True):
        with (OUT/name).open('x', encoding='utf8', newline='') as stream:
            stream.write(contents)
    print(json.dumps(dict(files={name:pin(OUT/name) for name in names}, calls=4680, blocks=156,
                          failed_controls=failed, rescored=False, admitted=False)))


if __name__ == '__main__':
    main()
