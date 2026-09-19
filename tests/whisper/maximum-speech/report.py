"""Render only closed, verified maximum-speech observations; never run inference."""
from pathlib import Path
import argparse
from prepare import sha, read, write_new, pin


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows',type=Path,required=True)
    parser.add_argument('--amd',type=Path,required=True)
    parser.add_argument('--markdown',type=Path,required=True)
    parser.add_argument('--json',type=Path,required=True)
    args = parser.parse_args()
    assert not args.markdown.exists() and not args.json.exists()
    windows,amd = args.windows.resolve(),args.amd.resolve()
    for base in (windows,amd):
        receipt = read(base/'closed.json')
        assert receipt['closed'] and receipt['all_owned_processes_terminal']
        assert {p.relative_to(base).as_posix() for p in base.rglob('*') if p.is_file()} == set(receipt['files'])|{'closed.json'}
        for name,expected in receipt['files'].items():
            assert pin(base/name) == expected,name
    w,a = read(windows/'audit.json'),read(amd/'audit.json')
    assert w['passed'] and a['passed'] and a['windows_audit_sha256'] == sha(windows/'audit.json')
    assert read(amd/'closed.json')['windows_closed_sha256'] == sha(windows/'closed.json')
    observations = dict(schema=1,windows=w,amd=a,windows_closed_sha256=sha(windows/'closed.json'),
        amd_closed_sha256=sha(amd/'closed.json'),product_source=read(windows/'frozen.json')['product_source'],
        runner_sha256=sha(windows/'bin/RecordingReplay.dll'),core_sha256=sha(windows/'bin/Lokad.Onnx.dll'),
        data_sha256=sha(windows/'bin/Lokad.Onnx.Data.dll'),reporter_sha256=sha(Path(__file__)))
    write_new(args.json,observations)
    lines = ['# Whisper: full ten-minute speech API qualification','',
        'The public recording API completes the retained 600-second speech request on Windows and AMD, including an exact repeat. All native application decisions, ownership, refusal/recovery and short-API checks pass. This is finite resource qualification using repeated English speech; natural long-conversation accuracy and full managed tensor agreement remain open.','',
        '## Workload and decisions','',
        'The input cycles the previously retained 69.455-second connected fixture and truncates at 9,600,000 mono 16 kHz samples. English is explicit; limits are 444 new tokens per window and 256 windows. Product source is `8732831`; both hosts use identical core, Data and runner DLLs.','',
        f"The native oracle completes 600 seconds in {w['rows'][0]['windows']} windows. An independent audit reconstructs every one of {w['native_steps']:,} greedy choices from full logits; all {w['native_arrays']:,} arrays containing {w['native_values']:,} float32 values are retained and hash-verified. These are native evidence arrays, not a managed intermediate tensor parity claim.",'',
        'Each managed process makes six recording calls: maximum speech and its repeat, empty input, maximum silence and two concurrent silence requests. It also checks ten refusals/cancellation recovery cases and one existing short transcription. Native, Windows and AMD tokens, text, timestamp segments, seek advances, stop and no-speech decisions agree exactly. Held outputs and caller inputs remain unchanged.','',
        '## Finite observations','',
        'Inference inherits logical CPU 2 before startup; supervisors use CPU 0. Windows uses i7-14700KF / .NET 10.0.12; AMD uses EPYC 9V74 / .NET 10.0.8. All times below cover the complete public API request, with features, inference, decoding and output ownership inside the stopwatch; loading and external validation are outside. These two sequential requests are resource/repeat observations, not a calibrated latency benchmark.','',
        '| Host | First request seconds | Repeat seconds | Windows / segments / tokens | Sampled process-group peak GB | Minimum system available GB |',
        '|---|---:|---:|---:|---:|---:|']
    for label,data,res in [('Windows',w,next(r for r in w['resources'] if r['phase']=='managed')),('AMD',a,a['resources'])]:
        r0,r1=data['rows']
        lines.append(f"| {label} | {r0['seconds']:.6f} | {r1['seconds']:.6f} | {r0['windows']} / {r0['segments']} / {r0['tokens']} | {res['peak_rss']/1e9:.6f} | {res['minimum_available_memory']/1e9:.6f} |")
    lines.extend(['',
        'GB are decimal. Peak RSS covers construction, both maximum requests, refusal/recovery and regression work; it is not a per-request allocation or a universal memory ceiling. Local guards were 16 GiB / 7,200 seconds / 1 GiB available; AMD guards 13.5 GiB / 3,600 seconds / 256 MiB available. Every sampled guard passes. Native generation includes array export and validation; its elapsed duration is not an ORT inference baseline. Use the [matched audio baselines](../../../BENCHMARK.md#audio-matched-microsoft-onnx-runtime-baselines) for dedicated comparisons.','',
        f"Maximum observed native confidence differences are {max(r['maximum_native_confidence_difference'] for r in w['rows']):.12g} on Windows and {max(r['maximum_native_confidence_difference'] for r in a['rows']):.12g} on AMD. Maximum cross-host difference is {max(r['maximum_windows_confidence_difference'] for r in a['rows']):.12g}. These remain diagnostics; no numerical threshold was fitted or changed.",'',
        '## Preserved failures and closure','',
        'The first local memory preflight refused before model load. A later attempt passed the same unchanged starting margin. The native worker then completed with exit 0, but its supervisor failed an immediate post-exit identity assertion. A separate recovery reverified all 1,917 arrays, independent native choices, immutable payload and complete resource samples after every recorded identity was absent. The original failure remains intact; inference was not rerun. Corrected supervision uses current PID existence plus birth identity and a bounded exit drain.','',
        'Six test methods pass, including 15 damaged application records, 11 damaged Windows resource records, 8 damaged AMD resource records, 6 runtime guard refusals and a real process birth/exit check. All native/managed supervisors and observed workers are terminal; every successful writer is closed. See [lane instructions](README.md) for reproduction and scope. CLI maximum speech is outside this API-only qualification.','',
        '## Evidence identities','',
        f"- Windows closure: `{observations['windows_closed_sha256']}`.",
        f"- AMD closure: `{observations['amd_closed_sha256']}`.",
        f"- Native manifest: `{w['native_sha256']}`.",
        f"- Native terminal recovery: `{sha(windows/'native-terminal-recovery.json')}`.",
        f"- Core: `{observations['core_sha256']}`.",
        f"- Data: `{observations['data_sha256']}`.",
        f"- Runner: `{observations['runner_sha256']}`.",'',
        'All request durations, confidence observations, resource counts, audit identities and terminal process births are in [the accompanying observations](observations-20260919.json). Local raw evidence is under `artifacts/whisper-maximum-speech-20260919` and `artifacts/whisper-maximum-speech-amd-20260919`.'])
    args.markdown.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Wrote closed observations and report.')


if __name__ == '__main__':
    main()
