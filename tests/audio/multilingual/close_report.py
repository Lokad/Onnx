"""Close audited evidence once, then publish every score and transcript."""
from pathlib import Path
import argparse
import json
import subprocess
import time
from audit import audit
from common import pin,read,sha,write_new


def close(base):
    root=Path(__file__).resolve().parents[3]
    assert not (base/'closed.json').exists()
    value=read(base/'audit.json')
    assert value['execution_passed'] and value==audit(base)
    assert value['auditor_sha256']==sha(base/'runtime-source/audit.py')
    prior=root/'artifacts/asr-multilingual-v2-20260920'
    failed=read(prior/'failed-closed.json')
    assert failed['closed'] and not failed['execution_passed']
    assert sha(prior/'failed-closed.json')==read(base/'preparation-reuse.json')['failed_closed_sha256']
    for name,wanted in failed['files'].items():assert pin(prior/name)==wanted,name
    preparation=root/'artifacts/asr-multilingual-20260920'
    record=dict(schema=1,closed=True,execution_passed=True,application_passed=value['application_passed'],
        closed_at=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        all_owned_processes_terminal=True,terminal_processes=value['terminal_processes'],audit_sha256=sha(base/'audit.json'),
        failed_attempt_receipt=pin(prior/'failed-closed.json'),
        preparation_provenance={p.relative_to(root).as_posix():pin(p) for p in sorted(preparation.rglob('*')) if p.is_file()},
        files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()},
        sources={p.relative_to(root).as_posix():pin(p) for p in sorted(Path(__file__).parent.iterdir()) if p.is_file()})
    write_new(base/'closed.json',record)
    print('Closed',len(record['files']),'files; receipt',sha(base/'closed.json'))


def escape(text):
    return text.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('|','\\|').replace('\n',' ')


def report(base,destination):
    root=Path(__file__).resolve().parents[3]
    names=['results-20260920.md','observations-20260920.json','transcripts-20260920.md']
    assert all(not (destination/name).exists() for name in names)
    receipt=read(base/'closed.json');assert receipt['closed'] and receipt['execution_passed'] and receipt['all_owned_processes_terminal']
    assert {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()}==set(receipt['files'])|{'closed.json'}
    for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
    for name,wanted in receipt['sources'].items():assert pin(root/name)==wanted,name
    value=read(base/'audit.json');assert sha(base/'audit.json')==receipt['audit_sha256']
    value=dict(value,closed_sha256=sha(base/'closed.json'),reporter_sha256=sha(Path(__file__)),
               protocol_source=read(base/'frozen.json')['source_commit'],failed_attempt_receipt=receipt['failed_attempt_receipt'])
    write_new(destination/names[1],value)
    lines=['# Multilingual and controlled-noise ASR results','',
        'The fixed Windows comparison completes all **164 requests**: forty cases and one repeat for each recognizer/engine. '
        'Execution, input/output ownership, repeat, process and resource checks pass. '
        f"The complete native/managed application gate **{'passes' if value['application_passed'] else 'fails'}**. "
        'Every recognition error and application disagreement is retained below and in the linked records.','',
        'The sample is twenty FLEURS read recordings in English, French, German, Spanish and Italian (four per language), '
        '232.06 seconds total. Each has a deterministic 10 dB additive-noise counterpart, giving forty cases and '
        '464.12 supplied audio seconds per recognizer/engine, excluding the repeat. This is a small fixed diagnostic; '
        'it is not the full FLEURS benchmark, conversational speech or a natural-noise corpus. Parallel translations '
        'are correlated, speaker identities are unavailable and model training exposure is unknown.','',
        '## Human-reference scores','',
        'WER and CER divide total word/character edit errors by total reference units, with normalized spaces included '
        'in CER. Independent edit distance agrees with JiWER 4.0.0. The [prospective protocol](README.md) preserves '
        'accents, does not verbalize digits, and fixes every selection/noise/normalization rule. '
        'Whisper receives the declared language; Parakeet detects it automatically. No accuracy threshold or model ranking is fitted.','',
        '| Recognizer | Language | Input | Words | Lokad word errors | ORT word errors | Lokad WER | ORT WER | Lokad CER | ORT CER |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    labels=dict(en_us='English',fr_fr='French',de_de='German',es_419='Spanish',it_it='Italian',all='All five')
    for family,model in value['models'].items():
        for group in model['groups']:
            m,n=group['managed'],group['native']
            assert m['reference_words']==n['reference_words']
            lines.append(f"| {family.title()} | {labels[group['locale']]} | {group['condition']} | {m['reference_words']} | {m['word_errors']} | {n['word_errors']} | {100*m['word_error_rate']:.4f}% | {100*n['word_error_rate']:.4f}% | {100*m['character_error_rate']:.4f}% | {100*n['character_error_rate']:.4f}% |")
    lines+=['','Each language row has four recordings; each all-language row has twenty. Clean/noisy versions share labels '
        'and are scored separately. [Every reference and both transcripts](transcripts-20260920.md) includes per-case errors. '
        '[Complete records](observations-20260920.json) retain normalized text, error counts, token decisions and source identities.','',
        '## Native agreement','']
    for family,model in value['models'].items():
        mismatches=[r['name'] for r in model['cases'] if not r['application_matches']]
        lines.append(f"{family.title()}: {40-len(mismatches)}/40 distinct cases match complete public decisions; the first-case repeat is stable within each engine. "
                     f"Full 41-request application gate: **{'PASS' if model['application_passed'] else 'FAIL'}**.")
        if mismatches:lines+=['','Disagreeing cases: '+', '.join('`'+name+'`' for name in mismatches)+'.']
        lines.append('')
    lines+=['Agreement includes tokens, text and stop/no-speech decisions, plus Parakeet frame indices, durations, '
        'encoded length and decoder-call count. Managed confidence scalars are retained without claiming native bit equality. '
        'The two engines compute independently from PCM; neither consumes the other\'s results. These application checks '
        'do not change the existing failed full-pipeline tensor gates or their 1e-4 tolerance.','',
        '## Finite timing and resource observations','',
        'Intel i7-14700KF, Windows logical CPU 2 inherited before startup; supervisor CPU 0. .NET 10.0.12 uses normal '
        'runtime settings and qualified product defaults. ORT 1.29.0 uses one intra/inter-op thread, sequential execution, '
        'all graph optimizations and no thread spinning. Four fresh workers run in the fixed native/managed order. '
        'The active workstation has unrelated activity. These are single-pass observations, with no warmed performance '
        'estimate or confidence claim; use the separate repeated [audio baselines](../../../BENCHMARK.md#audio-matched-microsoft-onnx-runtime-baselines).','',
        '| Worker | 40 complete PCM requests (s) | Constructor (s) | First-case repeat (s) | Process duration (s) | Peak sampled group RSS (GB) | Minimum system available (GB) | Foreign CPU fraction |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for resource in value['resources']:
        run=read(base/'run'/resource['name']/'result.json')
        lines.append(f"| {resource['name']} | {sum(r['seconds'] for r in run['cases'][:40]):.6f} | {run['constructor_seconds']:.6f} | {run['cases'][40]['seconds']:.6f} | {resource['seconds']:.6f} | {resource['peak_rss']/1e9:.6f} | {resource['minimum_available']/1e9:.6f} | {resource['foreign_cpu_fraction']:.6f} |")
    lines+=['','Request timing includes features, neural inference, decoding and owned results. Model loading, file access '
        'and external validation are outside that timer. No calls are removed, no forced GC is used, and retained outputs '
        'are checked after every request. All 20 GiB RSS, 3,600-second and 1 GiB available-memory guards pass; '
        'managed creation requires 20 GiB available, with at most ten minutes of recorded waiting. Snapshot foreign '
        'CPU fractions miss some exited/short-lived activity. GB is decimal here; guard GiB is binary.','',
        '## Reproduction and retained failures','',
        'Data: Google [FLEURS](https://huggingface.co/datasets/google/fleurs), revision '
        '`70bb2e84b976b7e960aa89f1c648e09c59f894dd`, CC-BY-4.0. [Dataset pins](dataset.json) bind all five original test '
        'files and the model card. Full metadata selection precedes recognition, and SoundFile/ffmpeg decode identical PCM. '
        'Both clean and noisy signals receive the same gain to avoid clipping; the signal auditor independently verifies '
        '10 dB SNR within 0.001 dB. German has only dataset gender label 0, so it does not have balanced labels or established speaker diversity.','',
        'The initial metadata policy requiring both gender labels failed before audio transforms or inference. '
        'The corrected available-label policy retains all five languages and four distinct sentence IDs each. '
        'A later first inference attempt stopped after nine native requests because a Windows reader denied atomic status '
        'replacement. That failure and all terminated process identities remain retained. The corrected writer retries '
        'that specific sharing failure for at most one second; an actual Windows file-lock test checks transient recovery '
        'and bounded refusal. The rerun uses identical input/replay bytes and unchanged selection, scoring and decoding. '
        'Nine tooling test methods pass; no product/default/tolerance changed.','',
        f"Frozen runtime source: `{value['protocol_source']}`. Qualified product source: `087e280b5ea0a6a610399ccffd1a1e5668def10e`; "
        'core `187de61ad8f034b9b7ad2fb3490358443fa84334204720e81bc3546a31f3c8d4`; '
        'Data `809242b58725c6ae47514cc3908ef59ffafae6be36bb6e2fba20144d9a975af5`. '
        'This newer qualified payload is distinct from the older matched audio timing binaries.','',
        'Raw inputs, full API results, samples, manifests and frozen source are under `artifacts/asr-multilingual-v3-20260920`. '
        'The auditor checks all 164 records, every case/clock/ownership/decision contract, source identities, resource '
        'samples and absence of every observed worker/supervisor birth. All writers are closed and single-use.','',
        f"Closed receipt: `{value['closed_sha256']}`. Failed inference receipt: `{value['failed_attempt_receipt']['sha256']}`. "
        f"Frozen manifest: `{value['frozen_sha256']}`.",'']
    with (destination/names[0]).open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines))
    lines=['# Every multilingual ASR transcript','',
        'References are FLEURS raw human transcripts (CC-BY-4.0). Errors use the fixed normalized policy, not raw '
        'punctuation/case differences. Complete token and normalized records are in [observations](observations-20260920.json).','']
    for family,model in value['models'].items():
        lines+=['## '+family.title(),'']
        for row in model['cases']:
            m,n=row['managed_metrics'],row['native_metrics']
            lines+=['### '+row['name'],'',
                '**Human:** '+escape(row['reference_text']),'',
                '**Lokad:** '+escape(row['managed']['text']),'',
                '**ORT:** '+escape(row['native']['text']),'',
                f"Word errors: Lokad {m['word_errors']}/{m['reference_words']}, ORT {n['word_errors']}/{n['reference_words']}. "
                f"Character errors: Lokad {m['character_errors']}/{m['reference_characters']}, ORT {n['character_errors']}/{n['reference_characters']}. "
                f"Complete application decisions match: {row['application_matches']}.",'']
    with (destination/names[2]).open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines))
    print('Reported every score and transcript; native application agreement:',value['application_passed'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=('close','report'))
    parser.add_argument('--artifact',type=Path,required=True);parser.add_argument('--destination',type=Path,default=Path(__file__).parent)
    args=parser.parse_args()
    if args.action=='close':close(args.artifact.resolve())
    else:report(args.artifact.resolve(),args.destination.resolve())
