"""Publish the closed full application comparison and every original timing clock."""
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-convolution-pointer-app-amd-20260923'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests/pyannote/convolution-pointer-app-amd'))
from protocol import TIMING_ROLES,pin,read,save
from admission import evaluate


def main():
    target=OUT/'application-20260923.md';assert not target.exists()
    closure=read(BASE/'closed.json');assert closure['passed']
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');assert analysis['performance']==evaluate(analysis['table'])
    payload=read(BASE/'payload.json');results=[];clocks=[];setups=[]
    for index,role in enumerate(TIMING_ROLES):
        value=read(BASE/'collected'/f'timing-{index:02}-{role}'/'output/result.json')
        results.append(value);setups.append(dict(process=index,role=role,seconds=value['setup_seconds']))
        for row in value['records']:
            clocks.append(dict(process=index,role=role,**{key:row[key] for key in ['name','pass','phase','start_ticks','end_ticks','frequency','seconds']}))
    assert len(clocks)==96 and sum(r['phase']=='warmup' for r in clocks)==24
    for rows,name in [(clocks,'application-clocks-20260923.csv'),(setups,'application-setup-20260923.csv')]:
        with (OUT/name).open('x',encoding='utf8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    save(OUT/'application-observations-20260923.json',dict(closure=pin(BASE/'closed.json'),payload=pin(BASE/'payload.json'),analysis=analysis,
        clocks=pin(OUT/'application-clocks-20260923.csv'),setup=pin(OUT/'application-setup-20260923.csv')))
    full,=[r for r in analysis['table'] if r['audio_seconds']==30]
    reduction=100*(1-full['ratios_to_selected']['candidate'])
    decision=analysis['performance'];bad_controls=[r for r in decision['controls'] if not r['passed']];bad_gains=[r for r in decision['gains'] if not r['passed']]
    lines=['# M28 complete Pyannote application comparison','',
        '**Admitted for root integration.**' if decision['admitted'] else '**Not admitted for root integration.**',
        f'Complete 30-second dialogue latency is {full["selected"]["seconds"]:.9f} s for the',
        f'current product, {full["candidate"]["seconds"]:.9f} s for the candidate and',
        f'{full["ort"]["seconds"]:.9f} s for Microsoft ONNX Runtime.',
        f'The candidate changes latency by {-reduction:+.4f}%; candidate / ORT is {full["ratios_to_ort"]["candidate"]:.9f}.',
        f'{len(bad_controls)} of twelve repeatability controls and {len(bad_gains)} of four speed gates fail.','',
        '| Complete request | Current seconds | Candidate seconds | Microsoft ORT seconds | Candidate / ORT | Candidate / current |',
        '|---|---:|---:|---:|---:|---:|']
    for row in analysis['table']:
        lines.append(f'| {row["name"]} | {row["selected"]["seconds"]:.9f} | {row["candidate"]["seconds"]:.9f} | {row["ort"]["seconds"]:.9f} | {row["ratios_to_ort"]["candidate"]:.6f} | {row["ratios_to_selected"]["candidate"]:.6f} |')
    lines+=['','AMD EPYC 9V74 CPU2, .NET 10.0.8 and ORT 1.29.0. Native ORT uses',
        'one intra/inter-op thread, sequential execution, full graph optimization',
        'and no spinning. Complete pinned native dependencies are verified.',
        'There are no managed numerical overrides or profilers.','',
        'Six fresh processes run current, candidate, ORT, ORT, candidate, current.',
        'One warmup and three measured passes over four fixtures retain all',
        '96 requests: 24 warmups and 72 measurements. Setup is recorded separately.',
        'Means reconstruct all six measured calls per fixture/role from raw ticks.',
        'Timing includes frontend, inference and owned public results. The full',
        'dialogue uses 21 overlapping windows; the separate ten-second crops',
        'are different workloads, and their times are not added to reconstruct it.','',
        'Frozen controls require process-mean max/min <=1.10 for the dialogue',
        'and <=1.20 for each crop, for every role. Candidate/current must be',
        '<=0.97 for the dialogue and <=1.05 for each crop. Every control and',
        'speed gate is mandatory. No samples were excluded or trimmed. These',
        'application gates were fixed before M28 component timing. The separate',
        'application parity target remains Lokad/ORT <=1.05.']
    if bad_controls or bad_gains:
        lines+=['','Failed criteria:','']
        for row in bad_controls:lines.append(f'- Repeatability, {row["role"]} / {row["name"]}: {row["process_ratio"]:.9f}, limit {row["limit"]:.2f}.')
        for row in bad_gains:lines.append(f'- Speed, {row["name"]}: {row["ratio"]:.9f}, limit {row["limit"]:.2f}.')
    meetings=analysis['results']['meetings-run']
    samples=sum(r['samples'] for r in analysis['resources']);peak=max(r['peak_rss'] for r in analysis['resources'])
    lines+=['','Fresh native conformance passes four Pyannote requests and twenty Parakeet',
        'clips. All 64 managed timing results equal fresh current results exactly,',
        'including across processes. Complete native timelines, centroids,',
        'transcripts, read-only inputs and held-output checks pass.','',
        'Both 600-second meetings (ES2004a and IS1009a) and the 30-second recovery',
        f'pass all original comparisons, with maximum centroid error {meetings["maximum_centroid_error"]:.12g}.',
        'Their diagnostic clocks are separate from the timing table. Previously',
        'closed full-product, Pyannote, Parakeet and shared/e5 qualifications are',
        'verified as prerequisites, without repeating their completed checks.','',
        f'All ten jobs and {samples:,} resource observations pass; peak owned RSS is',
        f'{peak:,} bytes. Every recorded owner is terminal. Process snapshots and',
        'foreign-CPU accounting pass, retaining the known limitation that snapshots',
        'can miss short-lived processes. Repeatability gates remain mandatory.','',
        f'Closure: `{pin(BASE/"closed.json")["sha256"]}`.',
        f'Payload: `{pin(BASE/"payload.json")["sha256"]}`.','',
        '[Every timing clock](application-clocks-20260923.csv),',
        '[every setup interval](application-setup-20260923.csv),',
        '[all controls, identities and resources](application-observations-20260923.json).','',
        'Artifacts: `artifacts/pyannote-convolution-pointer-app-amd-20260923`.']
    for role in ['selected','candidate']:
        lines+=['',f'{role.title()} Core: `{analysis["identities"][role]["Lokad.Onnx.dll"]["sha256"]}`.',
            f'Data: `{analysis["identities"][role]["Lokad.Onnx.Data.dll"]["sha256"]}`.']
    lines+=['','Actual root integration and normal-root qualification remain pending.' if decision['admitted'] else
        'The selected root is unchanged. No unchanged timing retry or integration follows this verdict.',
        'This campaign supplies no new matched Parakeet timing ratio.']
    target.write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(dict(admitted=decision['admitted'],dialogue=full,report=pin(target))))


if __name__=='__main__':main()
