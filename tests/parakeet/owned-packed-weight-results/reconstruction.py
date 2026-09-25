"""Publish the closed reconstruction diagnostic without changing application admission."""
import csv
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-owned-packed-weight-reconstruction-cost-amd-20260925'
REVISION = '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def exact(value):
    return Fraction(value['numerator'], value['denominator'])


def main():
    closure = read(BASE / 'closed.json')
    assert closure['passed'] and closure['analysis'] == pin(BASE / 'analysis.json')
    for name, wanted in closure['files'].items():
        assert pin(BASE / name) == wanted, name
    analysis = read(BASE / 'analysis.json')
    assert analysis['passed'] and analysis['diagnostic_only']
    assert not analysis['application_scored'] and not analysis['release_admitted'] and not analysis['new_variant_selected']
    assert (analysis['requests'], analysis['warmup'], analysis['measured'], analysis['feed_forward_calls']) == (320, 80, 240, 30720)
    assert analysis['application_closure']['sha256'] == '959d8d180db3431e1aeb5d7ab555fd9b77035ae84d4a8a678d2204cccc71eb48'
    summary = analysis['summary']
    usable = summary['usable_for_attribution']
    assert usable == closure['usable_for_attribution'] == all(c['passed'] for c in summary['controls'])
    assert len(summary['controls']) == 92 and not summary['overhead_subtracted']
    cases = read(BASE / 'bundle/spec.json')['cases']
    frames = {case['name']: case['expected']['encoded_frames'] for case in cases}
    table = summary['table']
    assert [r['name'] for r in table[:-1]] == list(frames) and table[-1]['corpus']
    rows = []
    for row in table[:-1]:
        n = frames[row['name']]
        record = dict(name=row['name'], frames=n, reconstructs=bool(n % 2 and n % 3))
        for role in ['selected', 'candidate']:
            for metric, value in row[role].items():
                record[role + '_' + metric + '_seconds'] = value['seconds']
        rows.append(record)
    partitions = []
    for reconstructs in [False, True]:
        chosen = [r for r in table[:-1] if bool(frames[r['name']] % 2 and frames[r['name']] % 3) == reconstructs]
        values = {}
        for role in ['selected', 'candidate']:
            values[role] = {}
            for metric in ['encoder', 'feed_forward', 'copy_y', 'math', 'scales']:
                value = sum(exact(row[role][metric]) for row in chosen)
                values[role][metric] = dict(seconds=float(value), numerator=value.numerator, denominator=value.denominator)
        partitions.append(dict(reconstructs=reconstructs, clips=len(chosen), names=[r['name'] for r in chosen], **values))
    assert [p['clips'] for p in partitions] == [13, 7]
    for role in ['selected', 'candidate']:
        for metric in partitions[0][role]:
            assert sum(exact(p[role][metric]) for p in partitions) == exact(table[-1][role][metric])
    source_path = 'onnxruntime/core/mlas/lib/sgemm.cpp'
    native = subprocess.run(['git', '-C', str(ROOT / 'external/onnxruntime'), 'show', REVISION + ':' + source_path],
                            check=True, stdout=subprocess.PIPE).stdout
    native_text = native.decode()
    assert 'MlasSgemmKernelLoop(A + k, pb, c, CountK, M, CountN, lda, ldc, alpha, ZeroMode);' in native_text
    assert 'A += lda * RowsHandled;' in native_text and 'CountM -= RowsHandled;' in native_text
    assembly_path = 'onnxruntime/core/mlas/lib/x86_64/FgemmKernelAvx512FCommon.h'
    assembly = subprocess.run(['git', '-C', str(ROOT / 'external/onnxruntime'), 'show', REVISION + ':' + assembly_path],
                              check=True, stdout=subprocess.PIPE).stdout
    assembly_text = assembly.decode()
    assert '.LProcessCountM1:' in assembly_text and 'ProcessCountM 1' in assembly_text
    assert 'cmp     r8,12' in assembly_text and 'ProcessCountM 5' in assembly_text
    managed = ROOT / 'artifacts/parakeet-owned-packed-weight-scope-source-20260925/source/src/Lokad.Onnx/TensorOps.OwnedPackedMatMul.cs'
    managed_text = managed.read_text()
    assert 'return CountedCopy(packed.ToDenseTensor(), options.CopyReporter);' in managed_text
    assert 'MathOps.ShortWideMultiplyRemainder(1, n, k, x + rows * n, original, output + rows * k);' in managed_text
    source_evidence = dict(ort_revision=REVISION, ort_path=source_path,
                           ort_source=dict(bytes=len(native), sha256=hashlib.sha256(native).hexdigest()),
                           ort_assembly_path=assembly_path,
                           ort_assembly=dict(bytes=len(assembly), sha256=hashlib.sha256(assembly).hexdigest()),
                           managed_source=dict(path=managed.relative_to(ROOT).as_posix(), **pin(managed)),
                           every_native_row_leaf_directly_observed=False)
    paths = [OUT / ('reconstruction-20260925' + suffix) for suffix in ['.json', '.csv', '.md']]
    assert not any(path.exists() for path in paths)
    controls_passed = sum(c['passed'] for c in summary['controls'])
    text = '# Parakeet: measured cost of dense weight reconstruction\n\n'
    text += ('The diagnostic passes all repeatability controls.\n' if usable else
             '**Repeatability failed; these clocks do not qualify for quantitative attribution.**\n')
    text += f'{controls_passed}/92 controls pass across four fresh processes. Product binaries are unchanged.\n\n'
    text += '| Twenty-clip diagnostic | Selected M73 seconds | Candidate M76 seconds |\n|---|---:|---:|\n'
    for metric, label in [('encoder', 'Enclosing encoder'), ('feed_forward', 'Complete feed-forward groups'),
                          ('copy_y', 'Existing reconstruction stage'), ('math', 'Feed-forward Math stages'), ('scales', 'Separate scale nodes')]:
        row = table[-1]
        text += f"| {label} | {row['selected'][metric]['seconds']:.6f} | {row['candidate'][metric]['seconds']:.6f} |\n"
    text += '\nThe stage rows are components of complete feed-forward time; do not add them to it.\n'
    text += '\n| Previously established path | Clips | Selected feed-forward seconds | Candidate feed-forward seconds | Candidate reconstruction seconds |\n|---|---:|---:|---:|---:|\n'
    for part in partitions:
        label = 'Reconstruction' if part['reconstructs'] else 'No reconstruction'
        text += f"| {label} | {part['clips']} | {part['selected']['feed_forward']['seconds']:.6f} | {part['candidate']['feed_forward']['seconds']:.6f} | {part['candidate']['copy_y']['seconds']:.6f} |\n"
    failed = [c for c in summary['controls'] if not c['passed']]
    if failed:
        text += '\nFailed controls:\n\n'
        for c in failed:
            text += f"- {c['role']} / {c['name']} / {c['metric']}: {c['ratio']:.6f} > {c['limit']:.2f}.\n"
    text += '''
The original CopyY boundary includes dense reconstruction, allocation and
preparation before Math. It is not solely a memory-bandwidth measurement.
There are 609 full 16 MiB reconstructions per candidate corpus, on seven
previously identified sequence lengths. Across warmup and measured requests,
all 4,872 expected CopyY stages occur; every other target has exactly zero.
All 320 requests preserve outputs, input hashes and the prior traffic counts.
The 96 matrix products and their 48 scale nodes are counted exactly once.

ORT's matching source passes prepared B to its row loop and advances A and C
while consuming the remaining rows. Its AVX-512 routine has explicit smaller-row
paths, including one row, using that prepared operand. Lokad's final-row routine takes
dense B, causing the reconstruction. The installed native routine was previously
matched byte-for-byte; each individual row leaf was not directly observed.
This is a specific implementation difference supported by source and counters.

These clocks cover the encoder and existing profiler stages, not complete
transcription. Each process has one warmup and three measured complete passes;
every measured pass and both processes carry equal weight. No clock is trimmed,
no observer overhead is subtracted, and no application saving is projected by
subtracting CopyY. Changes to cache behavior and allocation can affect other work.
Failed controls remain failures and permit no unchanged retry.

The complete application result remains 57.823506 s versus Microsoft ORT
39.245713 s, a 1.473371 ratio. Its 2.589784% gain over selected M73 misses the
original 3% requirement. This diagnostic does not change that admission,
the qualified repository product, or BENCHMARK.md. The two prior e5
repeatability failures also remain release blockers.

[Every clip](reconstruction-20260925.csv),
[complete clocks, controls and identities](reconstruction-20260925.json),
[original application verdict](application-20260925.md),
[prior shortfall partition](shortfall-20260925.md),
[installed ORT diagnosis](../ort-diagnosis-results/ort-kernels-20260924.md).
'''
    text += f'\nDiagnostic closure: `{pin(BASE / "closed.json")["sha256"]}`.\n'
    with paths[0].open('x', encoding='utf8') as stream:
        json.dump(dict(passed=True, diagnostic_only=True, usable_for_attribution=usable,
                       closure=pin(BASE / 'closed.json'), analysis=pin(BASE / 'analysis.json'),
                       products=analysis['products'], summary=summary, partitions=partitions,
                       source_evidence=source_evidence, resources=analysis['resources'],
                       application_admitted=False, release_admitted=False, new_variant_selected=False,
                       prior_failed_application_gate=analysis['prior_failed_application_gate'],
                       failed_release_controls=analysis['failed_release_controls'], publisher=pin(Path(__file__))),
                  stream, indent=2, allow_nan=False)
    with paths[1].open('x', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with paths[2].open('x', encoding='utf8', newline='\n') as stream:
        stream.write(text)
    print(json.dumps(dict(published=True, usable_for_attribution=usable, controls_passed=controls_passed,
                          corpus=table[-1], partitions=partitions), indent=2))


if __name__ == '__main__':
    main()
