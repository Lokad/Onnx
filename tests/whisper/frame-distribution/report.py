"""Close the verified saved-frame analysis and render its descriptive report."""
from pathlib import Path
import argparse, datetime, shutil
from analyze import ROOT, pin, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    base = parser.parse_args().artifact.resolve()
    assert not (base/'closed.json').exists()
    observations, verification = read(base/'observations.json'), read(base/'verification.json')
    assert verification['passed'] and verification['checked_difference_values'] == 241920000
    assert verification['observations'] == pin(base/'observations.json')
    assert verification['metrics'] == observations['metrics'] == pin(base/'frame-metrics.npz')
    assert verification['source'] == pin(Path(__file__).with_name('verify.py'))
    assert observations['source'] == pin(Path(__file__).with_name('analyze.py'))
    for name, expected in observations['inputs'].items():
        assert pin(ROOT/name) == expected, name
    labels = {'original_MM-NN': 'Original pipelines', 'engine_MM-NM': 'Same managed features', 'engine_MN-NN': 'Same native features'}
    lines = ['# Where the saved Whisper encoder differences occur — 2026-09-20', '',
        'On the twenty unique recordings, 97.26–97.54% of failing values in the original and same-input engine comparisons occur at positions whose pre-attention local input footprint is beyond the end of the recording. '
        'Every recording’s largest engine discrepancy is in that region. However, thousands of values also fail before the recording ends. **This does not justify excluding any encoder output or changing the numerical gate.**', '',
        'This is a descriptive analysis of the [closed full-corpus comparison](../input-cross-isolated/results-20260920.md), with no new inference. '
        'All 84 arrays and all six prior differences are retained. The table counts twenty unique recordings; the repeated first recording is separately included in the complete observations.', '',
        '| Comparison | Failed values before end | At end boundary | After end | Fraction of failures after end | Clips with failures before end |',
        '|---|---:|---:|---:|---:|---:|']
    for term, label in labels.items():
        regions = observations['unique20'][term]
        before, boundary, after = [regions[k]['failed_values'] for k in ['before_end', 'end_boundary', 'after_end']]
        cases = sum(r['terms'][term]['regions']['before_end']['failed_values'] > 0 for r in observations['rows'][:20])
        assert all(r['terms'][term]['worst']['region'] == 'after_end' for r in observations['rows'][:20])
        lines.append(f'| {label} | {before:,} | {boundary:,} | {after:,} | {100*after/(before+boundary+after):.4f}% | {cases}/20 |')
    regions = observations['unique20']['original_MM-NN']
    lines += ['', f"The three regions contain {regions['before_end']['values']:,}, {regions['end_boundary']['values']:,} and {regions['after_end']['values']:,} values respectively. "
        'The original pipeline failure rates within those regions are 0.05174%, 1.21338% and 1.15542%. '
        'Region counts therefore describe concentration, not equal-sized populations.', '',
        'Encoder position `j` has an input-time center at `320*j` samples (20 ms). The centered 400-sample Fourier window and two three-tap convolutions with strides one and two give a conservative local footprint `[320*j-520, 320*j+520)`. '
        'The regions place that footprint wholly before, across, or wholly after the original PCM end. Convolution attributes are checked against the pinned model; PCM lengths and arrays are checked against the original corpus receipt. '
        'This footprint describes the frontend and convolutions only: the clipping floor and encoder attention depend on the whole recording. It is neither a speech detector nor evidence that later frames are irrelevant.', '',
        'All differences use the original native-output denominator `max(1, abs(NN))` and the unchanged `1e-4` threshold. '
        'Both same-input comparisons still fail before the recording end: their maxima there are `0.00241367519` on managed features and `0.00252671540` on native features. '
        'The original-pipeline maximum before the end is `0.00218594074`. No numerical acceptance or accuracy preference between FP32 engines follows.', '',
        'The next diagnostic should retain three explicitly selected cases: `121-121726-0000` (largest before-end discrepancy), '
        '`1089-134686-0002` (largest original/managed-feature discrepancy, frame 564/channel 1116), and '
        '`908-157963-0000` (largest native-feature engine discrepancy, frame 978/channel 988). '
        'Selection is post-hoc and diagnostic. Trace residual and normalization boundaries on identical inputs, retaining an unmodified-output bridge and any instrumentation-induced differences before attributing a kernel. '
        'Any resulting change must subsequently pass the complete original corpus; these three cases cannot qualify it.', '',
        'An independent verifier reconstructs all 241,920,000 difference values, all 31,500 frame summaries for each of six terms, and all region/whole-corpus totals. '
        'It independently enumerates convolution taps to verify region geometry. First/repeated-case metrics are exact. No model execution, timing, product change or acceptance exception is involved.', '',
        f"Original closed receipt: `{observations['origin']['sha256']}`. Observations: `{pin(base/'observations.json')['sha256']}`. "
        f"Per-frame arrays: `{observations['metrics']['sha256']}`.",
        'Complete observations, all per-frame counts/maxima/squared errors, 152 bound input identities, verification and source snapshots are retained under '
        '`artifacts/whisper-frame-distribution-20260920`. The source arrays remain in their original closed artifact.', '']
    report = Path(__file__).with_name('results-20260920.md')
    assert not report.exists()
    report.write_text('\n'.join(lines), encoding='utf-8')
    snapshot = base/'source'; snapshot.mkdir()
    for path in Path(__file__).parent.iterdir():
        if path.is_file(): shutil.copyfile(path, snapshot/path.name)
    files = {p.relative_to(base).as_posix(): pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
    write(base/'closed.json', dict(passed=True, closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), files=files,
        report=dict(file=report.relative_to(ROOT).as_posix(), **pin(report)), scope='Verified descriptive analysis; original numerical failures unchanged'))
    print('Closed', len(files), 'files;', pin(base/'closed.json'))


if __name__ == '__main__':
    main()
