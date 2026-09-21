"""Add a verified complete e5 phase to the local benchmark document."""
import argparse
import datetime
import json

from contract import pin, read, write
from remote import BASE, ROOT
from verify_report import verify_markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['aa', 'compare'], required=True)
    phase = parser.parse_args().phase
    verification = read(BASE/(phase+'-verification.json'))
    assert verification['passed'] is True
    report = ROOT/'tests/e5/randomized-processes'/(phase+'-results.md')
    observations = report.with_name(phase+'-observations.json')
    assert pin(report) == verification['report'] and pin(observations) == verification['observations']
    analysis = read(observations)
    assert analysis['frozen'] == verification['frozen'] == pin(BASE/'frozen.json')
    assert verify_markdown(report.read_text(encoding='utf8'), analysis) == 20
    benchmark = ROOT/'BENCHMARK.md'
    original = benchmark.read_text(encoding='utf8')
    title = '### e5: independently randomized fresh processes ('+phase+')'
    assert title not in original
    retained = BASE/('benchmark-'+phase); retained.mkdir(exist_ok=False)
    (retained/'before.md').write_bytes(benchmark.read_bytes())
    statistical = verification['statistical_screen']; diagnostic = verification['diagnostic_screen']
    table = report.read_text(encoding='utf8').split('## Execute boundary\n\n', 1)[1].split('\n\n', 1)[0]
    section = '\n'.join([
        title, '',
        'AMD EPYC 9V74, logical CPU 2, .NET 10.0.8, Microsoft ORT 1.23.2. '
        'All 1,800 fresh workers and 334,080 measured calls pass raw-output and resource checks. '
        'Times below are mean public Execute/Run milliseconds; reset is outside this boundary.', '',
        'A uses current managed defaults. '+('C repeats those defaults in this A/A control.' if phase == 'aa'
          else 'C enables the optional fingerprint cache and final wide LayerNorm transform.'), '',
        table, '',
        f"The statistical screen **{'passes' if statistical else 'fails'}**; the observed-variance guard "
        f"**{'passes' if diagnostic else 'fails'}**. "
        + ('The fixed comparison may proceed.' if phase == 'aa' and statistical and diagnostic else
           'No comparison is authorized by this failed A/A.' if phase == 'aa' else
           'This report alone does not promote product defaults.'), '',
        'Intervals are approximate 95% family intervals for this finite campaign, conditional on '
        'independent assignments, no interference and large-sample regularity. '
        + ('' if diagnostic else 'The observed dominance guard fails, so the confidence interpretation is withheld. ')
        + f'The [complete report](tests/e5/randomized-processes/{phase}-results.md) retains request timings, '
        'all medians/tails, process diagnostics, allocations, GC and the original assignments. '
        'Lower ratios mean faster execution; none of these assumptions guarantees future parity.', '', ''])
    marker = '### e5:'
    position = original.index(marker)
    changed = original[:position]+section+original[position:]
    # Keep dated snapshots so a later result can update BENCHMARK.md without
    # invalidating this evidence receipt.
    benchmark.write_text(changed, encoding='utf8', newline='\n')
    (retained/'after.md').write_bytes(benchmark.read_bytes())
    assert section in benchmark.read_text(encoding='utf8')
    write(retained/'receipt.json', dict(passed=True, phase=phase,
          verification=pin(BASE/(phase+'-verification.json')), report=pin(report),
          before=pin(retained/'before.md'), after=pin(retained/'after.md'),
          utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(updated=True, phase=phase, receipt=pin(retained/'receipt.json'))))


if __name__ == '__main__':
    main()
