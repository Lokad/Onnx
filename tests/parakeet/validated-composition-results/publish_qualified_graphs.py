"""Publish the separate e5 successor and the fully sourced release graph summary."""
import json
from pathlib import Path
from publish_release import publish,csv_text,resources
from qualified_graphs import BASE,E5,GRAPH,pin,read,verified,derive


def main():
    proof,analysis=verified(BASE);actual,_=derive();assert actual==analysis
    e5_proof,e5=verified(E5);row=e5['performance']
    e5_lines=['# Focused e5 qualification with a fixed longer warmup','',
        '**Admission passes.**' if e5_proof['admitted'] else '**Admission fails.**','',
        '| Selected seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate / selected |',
        '|---:|---:|---:|---:|---:|',
        f"| {row['current']:.9f} | {row['candidate']:.9f} | {row['ort']:.9f} | {row['ratio']:.6f} | {row['candidate_over_current']:.6f} |",'',
        'This one-case comparison follows the [runtime diagnosis](../../benchmarks/e5-runtime-diagnostic-results/report-20260924.md).',
        'Three fresh numerical processes precede six fresh timing processes in',
        'selected, candidate, ORT, ORT, candidate, selected order. Every timing',
        'process uses 1,200 fixed warmups and 180 measurements. All 8,289 calls,',
        '1,080 measurements and nine setup intervals remain. No call is trimmed.','',
        'The same selected/candidate binaries and all original graph execution,',
        'numerical, input and held-output checks remain. The compiled inspection',
        'preserves 65/66 original methods exactly; Main differs only in its two',
        'loop/count constants, with all flags, branches, locals and exceptions exact.',
        'No profiler, observation hook or runtime override is active.','',
        f"All {sum(c['passed'] for c in row['controls'])}/3 repeatability controls pass (max/min <=1.10).",
        f"Candidate/selected <=1.05: {row['regression_passed']}. Candidate arrays exactly match selected and fresh ORT scaled error is <=1e-4.",'',
        resources(e5),'',
        '[Every clock](e5-warmed-clocks-20260924.csv), [all setups](e5-warmed-setups-20260924.csv),',
        '[full observations](e5-warmed-observations-20260924.json).',
        'The [original failed comparison](graphs-20260924.md) remains unchanged.','',
        'Closure: `'+pin(E5/'closed.json')['sha256']+'`.']
    lines=['# Composed release: qualified graph comparisons','',
        '**All eight cases pass qualification.**' if proof['admitted'] else '**Qualification remains incomplete.**','',
        '| Case | Selected seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Warmups per process |',
        '|---|---:|---:|---:|---:|---:|']
    for result in analysis['performance']:
        lines.append(f"| {result['key']} | {result['current']:.6f} | {result['candidate']:.6f} | {result['ort']:.6f} | {result['ratio']:.6f} | {result['warmups']} |")
    lines+=['','Seven cases use their complete retained M66 comparisons. Only 30-token e5',
        'uses the [separate diagnosed successor](e5-warmed-20260924.md), whose longer',
        'warmup was fixed before execution. This summary preserves both source',
        'campaigns, including the original failed e5 result; it does not relabel',
        'that campaign or select a subset of its measured calls.','',
        'Every case has three numerical processes followed by six fresh timing',
        'processes in selected/candidate/ORT/ORT/candidate/selected order, with',
        '180 measured calls per process. These selected complete cases contain',
        '41,112 calls, 8,640 measurements and 72 setups. All 45,801 calls across',
        'both original source campaigns remain retained.','',
        f"Repeatability: {sum(c['passed'] for r in analysis['performance'] for c in r['controls'])}/24 controls pass at max/min <=1.10.",
        f"Regression: {sum(r['regression_passed'] for r in analysis['performance'])}/8 gates pass at candidate/selected <=1.05.",'',
        'Both sources bind the same actual Core binaries: selected `672e5f30`,',
        'candidate `37c24375`. Every output meets its original native numerical',
        'bound, shape, finite, immutable-input and ownership checks. Candidate',
        'arrays match selected exactly. AMD EPYC 9V74 CPU2, .NET10.0.8, ORT1.29.0.',
        'The graph timer includes Reset/Execute/owned outputs for Lokad and',
        'session.run for ORT. Setup and validation remain separate.','',
        '[Complete qualified-case clocks](qualified-graph-clocks-20260924.csv),',
        '[setups](qualified-graph-setups-20260924.csv), [observations](qualified-graph-observations-20260924.json).',
        '[Original graph protocol](../validated-composition-graphs-amd/README.md),',
        '[e5 successor protocol](../../benchmarks/e5-warmed-qualification-amd/README.md).','',
        'Source closures: original `'+pin(GRAPH/'closed.json')['sha256']+'`; e5 `'+pin(E5/'closed.json')['sha256']+'`.',
        'Qualification closure: `'+pin(BASE/'closed.json')['sha256']+'`.']
    publish({'e5-warmed-20260924.md':'\n'.join(e5_lines)+'\n',
        'e5-warmed-clocks-20260924.csv':(E5/'clocks.csv').read_text(),
        'e5-warmed-setups-20260924.csv':csv_text(e5['setups']),
        'e5-warmed-observations-20260924.json':json.dumps(dict(closure=pin(E5/'closed.json'),**e5),indent=2)+'\n',
        'qualified-graphs-20260924.md':'\n'.join(lines)+'\n',
        'qualified-graph-clocks-20260924.csv':(BASE/'clocks.csv').read_text(),
        'qualified-graph-setups-20260924.csv':csv_text(analysis['setups']),
        'qualified-graph-observations-20260924.json':json.dumps(dict(closure=pin(BASE/'closed.json'),**analysis),indent=2)+'\n'})
    print(json.dumps(dict(passed=True,admitted=proof['admitted'],e5_admitted=e5_proof['admitted'])))


if __name__=='__main__':main()
