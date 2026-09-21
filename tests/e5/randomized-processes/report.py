"""Publish a complete collected e5 phase; no partial-data performance verdict."""
import argparse
import datetime
import json

from contract import pin, read, write
from evidence import audit_collected
from remote import BASE, ROOT


def interval(value):
    if not value['bounded']:
        return 'unbounded'
    return f"{value['ratio']:.4f} [{value['interval'][0]:.4f}, {value['interval'][1]:.4f}]"


def render(analysis, meta, state):
    phase = analysis['phase']; rows = analysis['timing']['results']
    passed = analysis['timing']['statistical_screen'] and analysis['diagnostic_screen']
    text = [f'# e5 independently randomized processes: {phase}', '',
            f"The prospective screens **{'PASS' if passed else 'DO NOT PASS'}**. All raw-output and resource checks pass; "
            f"the statistical screen is {'passing' if analysis['timing']['statistical_screen'] else 'failing'} and "
            f"the observed-variance diagnostic is {'passing' if analysis['diagnostic_screen'] else 'failing'}.", '',
            f"This phase retains **{analysis['measured_calls']:,} measured calls**, **{analysis['conditioning_calls']:,} conditioning calls** "
            'and 1,800 first calls in 1,800 fresh workers. No sample or tail is removed.', '',
            f"AMD EPYC 9V74, CPU2, .NET10.0.8, ORT1.23.2; product source `{meta['product_source']}`. "
            'All three engines use the same fixed model and inputs. A uses current managed defaults; '
            + ('C also uses current defaults in this A/A phase.' if phase == 'aa' else 'C enables fingerprint strings and the final wide LayerNorm transform.'), '',
            'Intervals are approximate 95% family intervals, conditional on independent randomized assignments, '
            'no interference and large-sample regularity. They describe the fixed campaign positions; they do not '
            'guarantee future performance. [Method and limitations](README.md) includes counterexamples.', '',
            f"Maximum native scaled output error: {analysis['maximum_native_error']:.9g}. The largest single-cohort share "
            'of observed contrast variation must be at most20%; this operational guard does not prove regularity '
            'of unobserved potential outcomes. Passing this report does not change product defaults.', '']
    summaries = {(r['case'], r['policy'], r['role']): r for r in analysis['summaries']}
    for boundary in ('execute', 'request'):
        text += [f'## {boundary.capitalize()} boundary', '',
                 '| Case | Policy | A ms | C ms | ORT ms | C/A interval | '+('A/ORT (descriptive)' if phase == 'aa' else 'C/ORT interval')+' |',
                 '|---|---|---:|---:|---:|---|---|']
        for row in rows:
            if row['boundary'] != boundary:
                continue
            means = {role: summaries[(row['case'], row['policy'], role)]['boundaries'][boundary]['mean_ms'] for role in ('A', 'C', 'N')}
            native = f"{means['A']/means['N']:.4f}" if phase == 'aa' else interval(row['contrasts']['C/N'])
            text.append(f"| {row['case']} | {row['policy']} | {means['A']:.6f} | {means['C']:.6f} | {means['N']:.6f} | {interval(row['contrasts']['C/A'])} | {native} |")
        text.append('')
    started = datetime.datetime.fromtimestamp(state['started'], datetime.timezone.utc).isoformat()
    ended = datetime.datetime.fromtimestamp(state['ended'], datetime.timezone.utc).isoformat()
    text += ['## Retained diagnostics', '',
             'The companion observations JSON contains every process mean, all-call median/p95/maximum, '
             'position counts and means, chronological halves, startup/first-call medians, '
             'allocations, GC counts and per-worker resources. Raw calls and before/after arrays remain retained. '
             'Reset is excluded from Execute and included in the enclosing request.', '',
             f'UTC interval: {started} to {ended}.', '',
             f"Frozen payload SHA256: `{analysis['frozen']['sha256']}`.",
             f"Collection SHA256: `{analysis['collection']['sha256']}`.",
             f"Core SHA256: `{meta['core_sha256']}`.", '',
             'The independent verifier must complete before this report can supply an A/A gate. '
             'Any failed screen stops this design; counts and assignments are not changed to obtain a pass.', '']
    return '\n'.join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['aa', 'compare'], required=True)
    args = parser.parse_args(); phase = args.phase
    output = ROOT/'tests/e5/randomized-processes'/(phase+'-results.md')
    observations = output.with_name(phase+'-observations.json')
    assert not output.exists() and not observations.exists() and not (BASE/(phase+'-analysis.json')).exists()
    collected = BASE/('collected-'+phase)
    analysis = audit_collected(collected, phase)
    write(BASE/(phase+'-analysis.json'), analysis)
    write(observations, analysis)
    with output.open('x', encoding='utf8', newline='\n') as stream:
        stream.write(render(analysis, read(collected/'frozen.json'), read(collected/('result-'+phase)/'identity.json')))
    write(BASE/(phase+'-report.json'), dict(analysis=pin(BASE/(phase+'-analysis.json')), report=pin(output), observations=pin(observations)))
    print(json.dumps(dict(phase=phase, raw_passed=True, statistical_screen=analysis['timing']['statistical_screen'],
                          diagnostic_screen=analysis['diagnostic_screen'])))


if __name__ == '__main__':
    main()
