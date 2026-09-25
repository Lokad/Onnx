"""Publish every complete-case clock and the unchanged application verdict."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-slice-dense-conversion-app-amd-20260925'
OUT=Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    proof=read(BASE/'closed.json');value=read(BASE/'analysis.json')
    assert proof['passed'] and value['passed'] and proof['analysis']==pin(BASE/'analysis.json')
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    performance=value['performance'];assert proof['admitted']==performance['admitted']
    assert len(performance['controls'])==63 and len(performance['gates'])==21
    assert (value['timing_requests'],value['warmup'],value['measured'])==(480,120,360)
    corpus,=[r for r in value['table'] if r['is_corpus']]
    assert corpus['audio_seconds']==213.265 and len(value['table'])==21
    gain=1-corpus['candidate']['seconds']/corpus['current']['seconds']
    rows=[]
    for row in value['table']:
        rows.append(dict(name=row['name'],audio_seconds=row['audio_seconds'],
            current_seconds=row['current']['seconds'],candidate_seconds=row['candidate']['seconds'],
            ort_seconds=row['ort']['seconds'],candidate_over_ort=row['ratios_to_ort']['candidate'],
            gain=1-row['candidate']['seconds']/row['current']['seconds']))
    outputs=[OUT/('application-20260925'+suffix) for suffix in ['.json','.csv','.md']]
    assert not any(p.exists() for p in outputs)
    with outputs[0].open('x',encoding='utf8') as stream:json.dump(dict(closure=pin(BASE/'closed.json'),**value),stream,indent=2,allow_nan=False)
    with outputs[1].open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    verdict='passes' if performance['admitted'] else 'fails'
    controls=sum(r['passed'] for r in performance['controls']);gates=sum(r['passed'] for r in performance['gates'])
    text=f'''# Positional-copy optimization: complete Parakeet comparison

**The candidate {verdict} the unchanged application admission.** Complete
transcription latency is {100*abs(gain):.3f}% {'lower' if gain>=0 else 'higher'} than the current release.
Repeatability controls: {controls}/63. Performance gates: {gates}/21.
Candidate latency is {corpus['ratios_to_ort']['candidate']:.6f} times Microsoft ORT.

| Twenty-clip corpus (213.265 s audio) | Seconds | Relative to ORT |
|---|---:|---:|
| Current release | {corpus['current']['seconds']:.6f} | {corpus['ratios_to_ort']['current']:.6f} |
| Slice-conversion candidate | {corpus['candidate']['seconds']:.6f} | {corpus['ratios_to_ort']['candidate']:.6f} |
| Microsoft ORT 1.29.0 | {corpus['ort']['seconds']:.6f} | 1.000000 |

Six fresh processes execute current, candidate, ORT, ORT, candidate, current.
Each runs one warmup and three measured passes over all 20 clips: 480 requests,
120 warmups and 360 measurements. Every raw clock contributes with equal
process weights. All original numerical, public-result, input-immutability
and held-output checks pass. This run uses the ordinary application consumers,
without a profiler, runtime override or pooled historical clock.

The fixed thresholds remain at least 3% corpus gain, no clip more than 5%
slower, corpus process repeatability at most 1.10 and per-clip at most 1.20
for each engine. The separate parity target is candidate/ORT at most 1.05.
All {sum(r['samples'] for r in value['resources']):,} resource observations pass;
peak owned RSS is {max(r['peak_rss'] for r in value['resources']):,} bytes.
Every process owner is terminal with code 0.

The [matched positional-copy profile](profile-20260925.md) motivated this one candidate;
the [full model qualification](models-20260925.md) preserves both instruction
modes and native bounds. [Every case](application-20260925.csv) and
[all controls, gates, identities, process clocks and resources](application-20260925.json)
remain available. An application admission still requires shared/e5, Pyannote,
graph, meeting, normal-root/full-suite and package qualification before product
integration or BENCHMARK.md changes. A failed admission retains its verdict.

Closure: `{pin(BASE/'closed.json')['sha256']}`.
Raw evidence: `artifacts/parakeet-slice-dense-conversion-app-amd-20260925`.
'''
    with outputs[2].open('x',encoding='utf8',newline='\n') as stream:stream.write(text)
    print(json.dumps(dict(admitted=performance['admitted'],gain=gain,candidate_over_ort=corpus['ratios_to_ort']['candidate'],controls=controls,gates=gates)))


if __name__=='__main__':main()
