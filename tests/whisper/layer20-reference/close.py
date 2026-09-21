"""Record the independently verified local diagnostic and preserve all evidence."""
import datetime
import json
import shutil
from protocol import *


def main():
    verification = read(BASE/'verification.json'); assert verification['passed'] is True and verification['audit'] == pin(BASE/'audit.json')
    audit = read(BASE/'audit.json'); assert audit['passed'] is True and all(absent(b) for b in audit['births'])
    spec = read(BASE/'manifest.json')
    for name, wanted in spec['files'].items():
        assert pin(ROOT/name) == wanted, name
    for name, wanted in spec['numerical_files'].items():
        assert pin(name) == wanted, name
    folder = Path(__file__).parent
    report = folder/'results-20260921.md'; data = folder/'observations-20260921.json'
    assert not report.exists() and not data.exists() and not (BASE/'closed.json').exists()
    write(data, dict(manifest=audit['manifest'], pairs=audit['pairs'], rows=audit['rows'], resources=audit['resources']))
    rows = [r for r in audit['rows'] if r['reference'] == 'numpy' and r['index'] == 11]
    table = ['| Request | Features | Cell | Total max | Inherited max | Local max | Local own-input failures |',
             '|---:|---|---|---:|---:|---:|---:|']
    for row in rows:
        m = row['metrics']
        table.append(f"| {row['request']} | {row['features']} | {row['cell']} | {m['total']['max_scaled']:.9g} | {m['inherited']['max_scaled']:.9g} | {m['local']['max_scaled']:.9g} | {m['local_own']['failed_values']} |")
    all_local = [r['metrics']['local_own'] for r in audit['rows']]
    failed = sum(m['failed_values'] > 0 for m in all_local)
    text = '\n'.join([
        '# Whole-layer Whisper reference decomposition — 2026-09-21', '',
        f"All **48 reference calls / 576 arrays** pass both-reference agreement at `1e-9`; the maximum scaled difference is "
        f"**{max(r['maximum'] for r in audit['pairs']):.9g}**. All reference-input cut outputs reproduce the retained "
        'complete-encoder layer-20 outputs from both reference routes within the same bound. Repeated cases reproduce exact bytes.', '',
        f"Across all twelve boundaries, **{failed}/768 own-input local comparisons** exceed `1e-4`, "
        f"with maximum **{max(m['max_scaled'] for m in all_local):.9g}**. Both independent references are included in this count. "
        'The earlier complete-encoder failures remain unchanged.', '',
        'For each existing FP32 output, total error equals the reference change caused by its incoming FP32 state '
        'plus the rounding residual of the whole layer on that state. All signed decompositions are checked per coordinate. '
        'The table shows final layer-20 outputs against the NumPy/SciPy route; the '
        '[complete observations](observations-20260921.json) include both routes and all boundaries.', '',
        *table, '',
        'Scaled maxima in the first three error columns share `max(1, abs(reference output on full-reference input))` '
        'as denominator. The final column compares local error against its own-input reference with the unchanged `1e-4` limit. '
        'Different maxima can occur at different coordinates. Local and inherited residuals can reinforce or cancel; '
        'their norms are not causal percentages.', '',
        'Eight requests cover the same three selected natural recordings and first-case repeat, each with both feature '
        'sources. Cell letters mean FP32 inference engine then incoming-state producer: M managed, N native. '
        'No new FP32 encoder/model execution occurred. This selected-case diagnostic is not a complete-corpus '
        'numerical qualification, an AMD result, or a performance benchmark.', '',
        'Both routes reuse the previously validated exact promotion of original FP32 weights. NumPy/SciPy interprets '
        'the original 48 nodes; ORT float64 sections use Python math.erf with optimizations disabled and one thread. '
        'Original operators, weights, selected cases, all padded frames and acceptance bounds are retained.', '',
        f"The independent chunked verifier checks all 768 complete decompositions and {verification['checked_scalars']:,} scalar metrics. "
        f"All {len(audit['births'])} recorded inference process identities are terminal. "
        f"Maximum sampled worker RSS: {max(r['peak_rss'] for r in audit['resources']):,} bytes. "
        f"Minimum available memory: {min(r['minimum_available'] for r in audit['resources']):,} bytes.", '',
        f"Frozen source `{spec['source']}`; manifest SHA256 `{audit['manifest']['sha256']}`. "
        'Artifact: `artifacts/whisper-layer20-reference-v2-20260921`. All raw arrays, telemetry and identities remain retained. '
        'The promoted weight file is an immutable hard link to the existing reference asset.', '',
        'The initial attempt stopped before its first resource sample because hidden Windows console creation '
        'violated the no-child guard. No reference call completed. Original sources and failure records remain '
        'preserved; a process-only probe reproduced the console child and verified detached hidden launch. '
        'The corrected campaign uses a separate artifact with unchanged cases, counts and numerical limits.', '',
        'See the prior [selected full-reference comparison](../trace-reference/results-20260920.md) and '
        '[same-input projection diagnostic](../natural-projection-reference/results-20260920.md) for their separate scopes.', ''])
    with report.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(text)
    snapshots = BASE/'closed-source'; snapshots.mkdir()
    for path in folder.glob('*.py'):
        shutil.copyfile(path, snapshots/path.name)
    files = {p.relative_to(BASE).as_posix(): pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    write(BASE/'closed.json', dict(passed=True, files=files, reports={rel(p): pin(p) for p in [report, data]}, births=audit['births'],
                                  utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(passed=True, closure=pin(BASE/'closed.json'), report=rel(report), local_failed_comparisons=failed)))


if __name__ == '__main__':
    main()
