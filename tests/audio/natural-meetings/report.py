"""Render closed natural-recording evidence and complete public outputs."""
from pathlib import Path
import argparse
from common import pin,read,write
from evidence import directory,profile


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True)
    p.add_argument('--destination',type=Path,default=Path(__file__).parent);a=p.parse_args()
    base=a.artifact.resolve();destination=a.destination.resolve()
    markdown=destination/'results-20260920.md';observations=destination/'observations-20260920.json'
    assert not markdown.exists() and not observations.exists()
    receipt=read(base/'closed.json');assert receipt['closed'] is True and receipt['execution_passed'] is True
    assert receipt['all_owned_processes_terminal'] is True and read(base/'verification.json')['passed'] is True
    assert {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}==set(receipt['files'])|{'closed.json'}
    for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
    assert pin(Path(__file__))==pin(base/'postprocessing-source/report.py')
    audit=read(base/'audit.json');resources=read(base/'resource-audit.json');manifest=read(base/'manifest.json')
    verification=read(base/'verification.json')
    values={f:{e:read(directory(base,d,f)/'worker/result.json') for e,d in [('ort','native'),('managed','managed')]} for f in manifest['schedule']}
    selected,linux_manifest,linux_frozen=profile(base,'native','whisper')
    linux=None if selected==base else dict(manifest=linux_manifest,frozen=linux_frozen,collection=read(selected/'collection.json'),
        dependency_check=read(selected/'dependency-check.json'),preparation=read(selected/'preparation.json'),
        original_windows_retry=read(base/'recovery-outcome.json'))
    result=dict(schema=1,closed_receipt=pin(base/'closed.json'),frozen=pin(base/'frozen.json'),runtime_source_commit=receipt['source_commit'],
        reporter=pin(Path(__file__)),manifest=manifest,labels=read(base/'labels.json'),audit=audit,resources=resources,
        verification=verification,results=values,native_linux=linux,source_files=receipt['sources'],evidence_files=receipt['files'],
        external_pins=receipt['external_pins'],remote_external_pins=receipt['remote_external_pins'])
    write(observations,result)
    names=dict(parakeet='Parakeet TDT 0.6B V3',whisper='Whisper Large V3 Turbo')
    engines=dict(ort='Microsoft ORT application',managed='Lokad.Onnx')
    status=lambda v:'PASS' if v else 'FAIL'
    recovery_note=('The original Windows native Whisper attempt failed the available-memory guard, and its single declared Windows retry refused before inference because stable memory headroom was unavailable. The tables use the separately declared Linux reference sequence. Both Windows attempts and the repeated first-meeting comparison remain visible below.\n\n' if linux else '')
    text=f'''# Natural ten-minute meeting ASR — 2026-09-20

Parakeet and Whisper each process two uninterrupted ten-minute AMI meetings and a thirty-second recovery, using Lokad.Onnx and an independently advancing native ORT reference. All requests completed: **{status(audit['all_requests_completed'])}**. Exact public decisions: **{status(audit['public_comparison_passed'])}**. Combined application qualification: **{status(audit['application_passed'])}**. Human word errors below are a separate observation, with no accuracy cutoff selected after recognition.

{recovery_note}\
The inputs are the first 600 seconds of ES2004a and IS1009a Mix-Headset audio, independently selected before inference as the first session-a at each ES/IS test site. Each contains four annotated speakers; labeled overlap is 52.71 and 55.47 seconds. Mono 16-kHz PCM16 is decoded without resampling, normalization, trimming or concatenation. A final request repeats the first thirty seconds of ES2004a and is excluded from accuracy aggregates.

Audio and manual annotations v1.6.2 are from the [AMI corpus](https://groups.inf.ed.ac.uk/ami/download/), CC BY 4.0. Attribution: Carletta et al., *The AMI meeting corpus: A pre-announcement* (2006). Selection and words-only timing crosschecks use the [pinned pyannote/BUT Speech@FIT setup](https://github.com/pyannote/AMI-diarization-setup/tree/67c2d539286e89f68952d5dcf83912bd9f01dfae).

## Human-reference accuracy

References include complete lexical words inside each fixed crop, retaining fillers, truncated spellings and original annotation text. Punctuation and non-word events are excluded. Sort by start, end, speaker letter and original XML index, then use the existing multilingual NFKC/casefold/alphanumeric/apostrophe normalization. This is a chronological mixed-speaker WER observation, not an official AMI ASR benchmark: overlapping speech makes a single reference order ambiguous. All separate word/speaker/time annotations remain in the observations.

ES2004a has 1,113 normalized reference words; IS1009a has 1,348. Each excludes one word crossing 600 seconds; thirteen and twenty truncated lexical records remain included. The recovery has 45 reference words but is not independently scored here. WER is (substitutions + deletions + insertions) / reference words. CER uses character edit distance. JiWER 4.0.0/RapidFuzz 3.14.6 and a separate dynamic-programming edit-distance implementation agree on every scored transcript.

| Model | Meeting | Engine | Reference words | Substitutions | Deletions | Insertions | WER | CER | Stop |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
'''
    for row in audit['scores']:
        text+=f"| {names[row['family']]} | {row['name']} | {engines[row['engine']]} | {row['reference_words']} | {row['substitutions']} | {row['deletions']} | {row['insertions']} | {100*row['word_error_rate']:.4f}% | {100*row['character_error_rate']:.4f}% | {row['stop_reason']} |\n"
    text+='\nAggregate rates sum error counts and reference lengths across the two meetings.\n\n| Model | Engine | Word errors / reference | WER | CER |\n|---|---|---:|---:|---:|\n'
    for row in audit['aggregates']:
        text+=f"| {names[row['family']]} | {engines[row['engine']]} | {row['word_errors']} / {row['reference_words']} | {100*row['word_error_rate']:.4f}% | {100*row['character_error_rate']:.4f}% |\n"
    text+='''
## Public-output agreement

Both engines use production recording defaults, with explicit English for Whisper and no reference-text prompt. Native Parakeet independently plans boundaries and crosschecks its decoder trajectory with pinned original onnx-asr methods. Native Whisper uses the unchanged qualified decoder and original timestamp/seek function bodies; its array observer checks finite float32 values without retaining giant intermediate tensors. Complete returned objects are retained.

Text, tokens, windows, boundaries, timestamps, seeks and stop decisions must match exactly. Whisper confidence values retain the existing finite/probability/skip-policy checks; their differences are diagnostic. A differing confidence value is not a newly relaxed full-tensor gate. Existing scaled `1e-4` intermediate numerical failures remain open. All inputs and earlier returned objects are checked after later calls.

| Model | Request | Exact public decisions | Reported field mismatches | Maximum Whisper confidence difference |
|---|---|---|---:|---:|
'''
    for row in audit['comparisons']:
        text+=f"| {names[row['family']]} | {row['name']} | {status(row['passed'])} | {row['mismatch_count']} | {row['maximum_observed_confidence_difference']:.12g} |\n"
    if audit['retry_repeat'] is not None:
        repeated=audit['retry_repeat']
        text+=f"\nThe native Whisper recovery repeats the first meeting after the original worker's memory stop. Its completed public result compared with the earlier preserved result: **{status(repeated['passed'])}**, {repeated['mismatch_count']} reported mismatches; maximum confidence difference {repeated['maximum_observed_confidence_difference']:.12g}. This repeated excerpt is not counted twice in accuracy.\n"
    text+='''
Every mismatch is preserved in the observations. A length mismatch reports the differing lengths without claiming unmatched elements were compared. Two meeting excerpts do not establish accuracy across AMI, languages or arbitrary conversations. No model parameter, segmentation rule, reference policy or tolerance was adjusted after inference.

## Timing and resources

Managed requests use qualified Core087e280/Dataf568132 on AMD EPYC 9V74 CPU 2, .NET 10.0.8. Native ORT 1.29.0 uses Windows i7-14700KF CPU 2 for Parakeet and AMD EPYC 9V74 CPU 2 for Whisper, with one intra-op/inter-op thread, sequential execution, all optimizations and disabled spinning. Models and PCM are identical. Whisper uses the pinned Transformers 5.16.1 NumPy frontend; the exact native dependencies and sources are in the observations.

These are single accuracy replays without matched warmups or a latency comparison protocol. Parakeet uses different hosts; Whisper uses the same AMD host. Native times include reference validation and Parakeet upstream trajectory crosschecks. They do not define an inference speed ratio and do not replace the [matched Windows baselines](../../../BENCHMARK.md#audio-matched-microsoft-onnx-runtime-baselines). Loading and file reads precede the call timers.

| Model | Request | Lokad on AMD seconds | ORT reference host | ORT application seconds | Lokad / ORT windows |
|---|---|---:|---|---:|---:|
'''
    for family in manifest['schedule']:
        for i,case in enumerate(manifest['cases']):
            m=values[family]['managed']['records'][i];n=values[family]['ort']['records'][i]
            text+=f"| {names[family]} | {case['name']} | {m['seconds']:.6f} | {'AMD Linux' if family=='whisper' else 'Windows'} | {n['seconds']:.6f} | {len(m['result']['windows'])} / {len(n['result']['windows'])} |\n"
    text+='\n| Model | Host / engine | Worker seconds | Peak sampled RSS GB | Minimum available GB | Samples | Foreign CPU fraction |\n|---|---|---:|---:|---:|---:|---:|\n'
    for row in resources['resources']:
        text+=f"| {names[row['family']]} | {row['host']} / {row['engine']} | {row['seconds']:.6f} | {row['peak_rss']/1e9:.6f} | {row['minimum_available']/1e9:.6f} | {row['samples']} | {row['accounting']['foreign_cpu_fraction']:.9f} |\n"
    text+=f'''
The four completed workers satisfy the fixed two-hour and 1-GiB available-memory guards. Windows requires 20 GiB available before each worker and permits less than 20 GiB group RSS; AMD requires 13 GiB before launch and permits less than 14 GiB RSS. Memory is sampled every half second; peaks are finite observations. Foreign CPU accounting uses process snapshot deltas divided by wall time and logical CPU count and can miss exited processes. Windows remained an active workstation.

## Evidence and reproduction

All actual process births are terminal. Independent input checks reproduce human annotations and original PCM hashes. Four local and two AMD input-only workers pass. Two initial native input preflights refused before child creation because available memory was below 20 GiB; distinct retries passed with unchanged limits. Four offline Whisper wrapper cases exactly reproduce retained qualified outputs; fourteen retained native recordings pass the borrowed validators, and eight damaged Whisper copies refuse. On this new evidence, validators refuse {sum(r['count'] for r in verification['refusals'])} damaged application records and {sum(r['count'] for r in resources['refusals'])} damaged resource records.

The artifact is `artifacts/asr-natural-meetings-20260920`, frozen source `{receipt['source_commit']}`, frozen SHA256 `{pin(base/'frozen.json')['sha256']}`. Final receipt SHA256 `{pin(base/'closed.json')['sha256']}` binds {len(receipt['files'])} files and external native/scorer identities. [Complete transcripts, edit alignments, public outputs, sources and evidence hashes](observations-20260920.json) are retained. [Reproduction instructions](README.md) document preparation, exact schedule, runners and limits. Existing numerical failures and the separate e5 performance objective remain unresolved.
'''
    failed=resources.get('failed_native_attempt')
    if failed:
        text+=f"\nThe original native Whisper worker **failed its resource guard** after {failed['seconds_before_stop']:.6f} seconds: available system memory fell to {failed['last_available']:,} bytes, below 1 GiB, while its peak sampled group RSS was {failed['peak_rss']:,} bytes. Only ES2004a completed; all original {failed['resource_samples']} samples and the partial output are preserved. The separately declared Windows retry required sixty consecutive seconds above the original 20-GiB launch requirement; it refused after its fifteen-minute preflight without creating an inference worker. A new Linux reference ran all three calls under the declared AMD limits, with identical models, inputs and decoding options. Exact generator/frontend bytes and a portable syntax-tree comparison prove the decoding bodies unchanged across Python versions. The report scores this complete sequence and separately compares its first meeting with the earlier Windows result. Parakeet and managed Whisper were not repeated. The failed original campaign remains failed.\n"
    if linux:
        text+=f"\nThe Linux reference is frozen at source `{linux_frozen['source_commit']}`, SHA256 `{pin(selected/'frozen.json')['sha256']}`. Its {len(linux_frozen['native_files'])} native dependency identities were checked remotely at collection and closure; wheel installation reports, the input-only smoke, runtime sources, raw outputs and failed preparation check are retained. [Linux continuation instructions](../natural-meetings-linux/README.md) describe the isolated CPU-only Torch installation and platform proof.\n"
    else:text+='\nEach inference schedule ran once.\n'
    with markdown.open('x',encoding='utf-8') as stream:stream.write(text)
    print('Rendered',len(audit['scores']),'human scores and',len(audit['comparisons']),'public comparisons.')


if __name__=='__main__':main()
