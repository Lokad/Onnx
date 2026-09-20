"""Render the complete closed meeting comparison without a fitted accuracy cutoff."""
from pathlib import Path
import argparse
from common import pin,read,write


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--destination',type=Path,default=Path(__file__).parent)
    args=parser.parse_args();base=args.artifact.resolve();destination=args.destination.resolve()
    markdown=destination/'results-20260920.md';observations=destination/'observations-20260920.json'
    assert not markdown.exists() and not observations.exists()
    receipt=read(base/'closed.json');assert receipt['closed'] is True and receipt['execution_passed'] is True
    assert receipt['all_owned_processes_terminal'] is True and read(base/'verification.json')['passed'] is True
    assert {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}==set(receipt['files'])|{'closed.json'}
    for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
    assert pin(Path(__file__))==pin(base/'postprocessing-source/report.py')
    audit=read(base/'audit.json');resources=read(base/'resource-audit.json');manifest=read(base/'manifest.json')
    values={e:read(base/f'process-{d}-run/worker/result.json') for e,d in [('ort','native'),('managed','managed')]}
    result=dict(schema=1,closed_receipt=pin(base/'closed.json'),frozen=pin(base/'frozen.json'),
        runtime_source_commit=receipt['source_commit'],reporter=pin(Path(__file__)),manifest=manifest,
        dataset=read(base/'inputs/dataset.json'),audit=audit,resources=resources,verification=read(base/'verification.json'),
        results=values,source_files=receipt['sources'],evidence_files=receipt['files'])
    write(observations,result)
    status='passes' if audit['public_comparison_passed'] else 'fails'
    text=f'''# Natural ten-minute meeting diarization — 2026-09-20

Both engines complete ES2004a and IS1009a, two uninterrupted ten-minute AMI meeting excerpts, followed by a thirty-second recovery request. Inputs and held outputs remain unchanged. Strict public-output compatibility **{status}** under the predeclared comparison. Human-label accuracy below is a separate observation, with no acceptance cutoff chosen after seeing the results.

Selection preceded inference: the lexicographically first session-a meeting in each of the ES and IS sites of the official test list, at setup revision `67c2d539286e89f68952d5dcf83912bd9f01dfae`. Each excerpt contains four annotated speakers. The first 600 seconds of each mixed-headset mono 16-kHz PCM16 WAV are used unchanged, including silence and overlap. Labeled overlap is 52.71/55.47 seconds and reference speaker time is 420.92/481.67 seconds. The recovery crop repeats the first thirty seconds of ES2004a and is excluded from accuracy aggregates.

Audio and manual annotations are from the [AMI corpus](https://groups.inf.ed.ac.uk/ami/download/), CC BY 4.0. Words-only references and evaluation regions use the [pinned pyannote/BUT Speech@FIT setup](https://github.com/pyannote/AMI-diarization-setup/tree/67c2d539286e89f68952d5dcf83912bd9f01dfae), whose setup code is Apache 2.0. Attribution: Carletta et al., *The AMI meeting corpus: A pre-announcement* (2006); Landini et al., *Bayesian HMM clustering of x-vector sequences (VBx) in speaker diarization* (2022).

## Accuracy against human annotations

Official pyannote.metrics 4.1/core 6.0.1 scores the full 600-second region, with zero collar, overlap included and optimal one-to-one speaker mapping. DER is (missed + false alarm + confusion) / reference speaker-seconds; overlap contributes each annotated speaker. All components below are speaker-seconds. An independent interval sweep and exact assignment reproduce every component within 1e-8 seconds. Ordinary timelines may overlap; exclusive timelines allow one active speaker.

| Meeting | Engine | Timeline | Reference | Correct | Missed | False alarm | Confusion | DER |
|---|---|---|---:|---:|---:|---:|---:|---:|
'''
    for row in audit['scores']:
        text+=f"| {row['name']} | {'Microsoft ORT application' if row['engine']=='ort' else 'Lokad.Onnx'} | {'Ordinary' if row['timeline']=='intervals' else 'Exclusive'} | {row['reference_speaker_seconds']:.6f} | {row['correct_speaker_seconds']:.6f} | {row['missed_speaker_seconds']:.6f} | {row['false_alarm_speaker_seconds']:.6f} | {row['confused_speaker_seconds']:.6f} | {100*row['diarization_error_rate']:.4f}% |\n"
    text+='\nAggregate DER sums components across the two meetings; it is not the mean of their percentages.\n\n| Engine | Timeline | Reference | Missed | False alarm | Confusion | DER |\n|---|---|---:|---:|---:|---:|---:|\n'
    for row in audit['aggregates']:
        text+=f"| {'Microsoft ORT application' if row['engine']=='ort' else 'Lokad.Onnx'} | {'Ordinary' if row['timeline']=='intervals' else 'Exclusive'} | {row['reference_speaker_seconds']:.6f} | {row['missed_speaker_seconds']:.6f} | {row['false_alarm_speaker_seconds']:.6f} | {row['confused_speaker_seconds']:.6f} | {100*row['diarization_error_rate']:.4f}% |\n"
    text+='''
## Public-output compatibility

Status, model-window count and speaker identities must agree exactly; ordered interval boundaries use the existing 1e-12 absolute tolerance and centroid coordinates use `abs(managed-native)/max(1,abs(native)) <= 1e-4`. Every field disagreement is retained in the linked observations. A length mismatch reports the differing array lengths without pretending unmatched coordinates were compared. The maximum below only covers aligned centroid arrays.

| Request | Pass | Reported field mismatches | Maximum aligned centroid scaled error | Lokad / ORT speakers |
|---|---|---:|---:|---:|
'''
    for i,row in enumerate(audit['comparisons']):
        text+=f"| {row['name']} | {row['passed']} | {row['mismatch_count']} | {row['maximum_centroid_error']:.12g} | {len(values['managed']['records'][i]['result']['speakers'])} / {len(values['ort']['records'][i]['result']['speakers'])} |\n"
    text+='''
This checks returned application results; it does not resolve the known intermediate filterbank numerical failures or certify all long intermediate tensors. Two excerpts do not establish general accuracy for AMI or arbitrary conversations. No speaker-count constraint, model parameter, timeline rule or numerical tolerance was adjusted for this evaluation.

## Execution and finite resource observations

Lokad.Onnx uses qualified Core087e280/Dataf568132 binaries on AMD EPYC 9V74, logical CPU 2, .NET 10.0.8. The complete native application uses Microsoft ORT 1.29.0, Torch/torchaudio 2.11.0+cpu, NumPy 2.2.4, SciPy 1.16.3 and pinned upstream pyannote 4.0.0 methods on Windows i7-14700KF, logical CPU 2. ORT uses one intra-op/inter-op thread, sequential execution, all graph optimizations and disabled spinning. Frontend, pooling and clustering are included alongside ORT neural graphs. Exact binaries, sources, models and flags are in the observations.

These hosts differ. Single-pass API times include features, neural inference, clustering and owned output construction; loading and external validation are excluded. They are descriptive accuracy-replay times and do not define a speed ratio or replace the matched Windows latency baselines in [BENCHMARK.md](../../../BENCHMARK.md#audio-matched-microsoft-onnx-runtime-baselines).

| Request | Lokad on AMD seconds | ORT application on Windows seconds | Lokad / ORT windows |
|---|---:|---:|---:|
'''
    for i,case in enumerate(manifest['cases']):
        m=values['managed']['records'][i];n=values['ort']['records'][i]
        text+=f"| {case['name']} | {m['seconds']:.6f} | {n['seconds']:.6f} | {m['result']['windows']} / {n['result']['windows']} |\n"
    text+='\n| Host | Worker seconds | Peak sampled group RSS GB | Minimum available GB | Resource samples | Foreign CPU fraction |\n|---|---:|---:|---:|---:|---:|\n'
    for row in resources['resources']:
        text+=f"| {'Windows native' if row['engine']=='native' else 'AMD managed'} | {row['seconds']:.6f} | {row['peak_rss']/1e9:.6f} | {row['minimum_available']/1e9:.6f} | {row['samples']} | {row['accounting']['foreign_cpu_fraction']:.9f} |\n"
    text+=f'''
Both schedules satisfy the frozen 8-GiB process-group RSS, 3,600-second and 1-GiB available-memory guards, with at least 8 GiB available before launch. Memory is sampled every half second and is a finite observation. Foreign CPU accounting uses snapshot deltas divided by wall time and logical CPU count; it can miss exited processes. Windows was an active workstation. No confidence or parity claim follows from these timings.

## Evidence and validation

All actual process births are terminal on both hosts. Input preparation independently verifies original PCM bytes and annotation coverage. The official and independent scorers agree on 64 fixed-seed synthetic overlap cases and all eight actual meeting timelines. Validators reject 19 damaged copies of each engine's real request evidence and 14 damaged copies of each resource record. All raw calls, resource samples, failed preparation attempts and collection identities remain preserved. The two initial input-only supervisors failed before child launch due to a psutil path-type requirement; their corrected distinct runs pass. The first freeze used an incorrect installed-package path and failed before inference; the corrected freeze used the actual distribution path. Neither inference schedule was repeated.

The artifact is `artifacts/pyannote-natural-meetings-20260920`. Frozen runtime source is `{receipt['source_commit']}`; runtime manifest SHA256 is `{pin(base/'frozen.json')['sha256']}`. Final receipt SHA256 is `{pin(base/'closed.json')['sha256']}`, binding {len(receipt['files'])} artifact files plus external native/scorer pins. [All public outputs, human scores, mismatches, sources and evidence hashes](observations-20260920.json) are retained. [Reproduction instructions](README.md) describe each single-use writer.
'''
    with markdown.open('x',encoding='utf-8') as stream:stream.write(text)
    print('Published',len(audit['scores']),'human-score rows and all',len(audit['comparisons']),'public comparisons.')


if __name__=='__main__':main()
