"""Publish the single matched profile with every prospective group member retained."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-slice-dense-conversion-profile-resume-amd-20260925'


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    target=OUT/'profile-20260925.json';markdown=OUT/'profile-20260925.md'
    assert not target.exists() and not markdown.exists()
    closed=read(BASE/'closed.json');assert closed['analysis']==pin(BASE/'analysis.json')
    assert closed['transfer']==pin(BASE/'capture-transfer.json')
    assert closed['collection']==pin(BASE/'capture-collected/capture-collection.json')
    analysis=read(BASE/'analysis.json');assert closed['passed']==analysis['passed']
    assert analysis['attribution_only'] and not analysis['application_gain_admitted'] and not analysis['overhead_subtracted']
    spec=read(BASE/'bundle/spec.json');group=analysis['positional']
    phases={k:dict(corpus_seconds=v['corpus_seconds'],phase_seconds=v['phase_seconds'],
        remainder_seconds=v['remainder_seconds'],call_counts=v['call_counts']) for k,v in analysis['phases'].items()}
    result=dict(passed=analysis['passed'],closure=pin(BASE/'closed.json'),analysis=pin(BASE/'analysis.json'),
        initial_failure=closed['initial'],completed_control_repeated=analysis['completed_control_repeated'],
        prepared=pin(BASE/'prepared.json'),current=spec['core'],candidate=spec['candidate_core'],
        observed_data=spec['data'],consumer=spec['consumer'],prospective_groups=analysis['prospective_groups'],
        requests=analysis['requests'],clips=analysis['clips'],frames=analysis['frames'],
        positional={k:v for k,v in group.items() if k!='all_nodes'},phases=phases,
        resources=analysis['resources'],all_node_rows_retained=len(group['all_nodes']),
        observer_overhead_included=True,application_speedup_claimed=False,release_unchanged=True,
        publisher=pin(Path(__file__)))
    with target.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    lines=['# Parakeet positional-copy attribution — 2026-09-25','',
        f"Prospective prediction **{'passed' if result['passed'] else 'failed'}**. "
        f"The complete34-node positional group takes {group['selected_seconds']:.6f}s with the current Core "
        f"and {group['candidate_seconds']:.6f}s with the candidate ({100*group['complete_group_gain']:.2f}% reduction). "
        f"{group['improved_kernels']}/24 projection kernels improve.",'',
        'These clocks include the same wall observer in both processes. They establish attribution; '
        'they do not establish application speedup or a new ORT ratio. The release and BENCHMARK.md remain unchanged.','',
        'The current process completed before an immediate memory preflight refusal. The candidate '
        'had not started; a separate process completed only that missing leg after idle memory '
        'recovered. The initial code1 and both complete raw collections remain retained.','',
        'Both processes executed all20 clips with one warmup and three measured passes:160 requests. '
        'Complete public results match exactly. All graph metadata, request/phase/node intervals, '
        'input identities, CPU placement, resource limits and foreign CPU checks pass. '
        'Shared positional preparation is counted once. Every other node remains in the retained analysis.','',
        '| Profile interval | Current seconds | Candidate seconds |','| --- | ---: | ---: |']
    for name in ['frontend','encoder','decoder']:
        lines.append(f"| {name} | {phases['selected']['phase_seconds'][name]:.6f} | {phases['candidate']['phase_seconds'][name]:.6f} |")
    for name,key in [('Outside graph calls','remainder_seconds'),('Full request, profiled','corpus_seconds')]:
        lines.append(f"| {name} | {phases['selected'][key]:.6f} | {phases['candidate'][key]:.6f} |")
    lines+=['','| Positional group member | Current seconds | Candidate seconds |','| --- | ---: | ---: |']
    for row in group['rows']:lines.append(f"| `{row['name']}` | {row['selected_seconds']:.9f} | {row['candidate_seconds']:.9f} |")
    lines+=['',f"Closure: `{result['closure']['sha256']}`. Report: `{pin(target)['sha256']}`.",'',
        f"All raw requests and {result['all_node_rows_retained']} node comparisons are retained under "
        '`artifacts/parakeet-slice-dense-conversion-profile-resume-amd-20260925`. '
        'M72’s failed component repeatability controls remain recorded; no component clock was used as an application result.','']
    with markdown.open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines))
    print(json.dumps(dict(passed=result['passed'],report=pin(target),markdown=pin(markdown))))


if __name__=='__main__':main()
