"""Publish complete verified timings while retaining the original disk failure."""
import datetime
import re
import shutil
from common import *
from storage_contract import replace_pending
from statistics_exact import suffix
from verify import checked_rows


def main():
    rows, checks = checked_rows(); audit = read(BASE/'audit.json'); receipt = read(BASE/'collected/collection.json')
    assert audit['counts'] == dict(referenced_conformance=40,warmup=80,measured=240)
    for name, expected in receipt['files'].items():
        assert pin(BASE/'collected'/name) == expected, name
    ssh(PRELUDE+'terminal(%r)\n' % receipt['births'])
    headline = rows[0]
    header = '| Workload | Lokad seconds | Microsoft ORT seconds | Lokad / ORT | Lokad RTF | ORT RTF |\n|---|---:|---:|---:|---:|---:|'
    def line(row):
        return '| '+('Whisper Large V3 Turbo, all 20 clips' if row['name']=='complete-corpus' else row['name'])+suffix(row)
    table = header+'\n'+'\n'.join(map(line,rows)); short = header+'\n'+line(headline)
    report = TOOLS/'results.md'; observations = TOOLS/'observations.json'
    assert not report.exists() and not observations.exists() and not (BASE/'closed.json').exists()
    resources = '\n'.join(f"| {o['name']} | {o['resource']['samples']} | {o['resource']['peak_rss']:,} | {o['resource']['min_available']:,} | {o['resource']['min_disk']:,} |" for o in audit['observations'])
    variation = '\n'.join(f"| {engine} | {float(headline[engine]['visits'][0]['mean']):.6f} | {float(headline[engine]['visits'][1]['mean']):.6f} |" for engine in ['managed','ort'])
    text = f'''# Matched AMD Whisper Large V3 Turbo baseline

All four fresh timing workers complete the fixed comparison on AMD EPYC 9V74,
logical CPU 2, .NET 10.0.8 and Microsoft ONNX Runtime 1.29.0. The unchanged qualified
product is `{audit['product_source'][:7]}`. Twenty clips contain {float(headline['audio_seconds']):.3f} seconds of audio.

{short}

Lower is faster. Lokad/ORT above one means Lokad takes longer. Times are the mean
total for the full corpus, including features, neural inference, greedy decoding
and owned output construction. Loading, file access and external validation are
excluded. Both engines pad clips to the same thirty-second encoder input.
These descriptive observations have no calibrated confidence or parity claim.

Fresh processes run managed, ORT, ORT, managed. Each executes twenty warmup and
sixty measured calls. All 320 calls are retained, with 240 measured and 80 warmup.
The two original complete twenty-call conformance gates are reused only after
rechecking every output, resource sample, frontend array and identity. No old
timing sample is reused. Every new request passes exact text/token/stop/no-speech
decisions, unchanged inputs, held outputs and repeated-result checks.

ORT uses the pinned Transformers NumPy frontend, CPUExecutionProvider, one
intra/inter-op thread, sequential execution, all graph optimizations and no
spinning. Every sampled inference thread stays on CPU 2. Managed execution has
normal collection and no LOKAD/DOTNET/COMPlus overrides. Existing full encoder/logit
numerical failures remain open; application equality is a separate result.

The [original disk failure](../whisper-amd/disk-failure-20260921.md) remains
preserved. This separately declared comparison places new evidence and temporary
files on `/dev/shm`, verifies its actual device differs from `/`, and streams
collection directly to local disk. It requires 3 GiB free before launch and 512 MiB
during execution. Original 3,600-second / 14 GiB RSS / 1 GiB available-memory worker limits,
13 GiB preflight RAM and 14,400-second campaign bound are unchanged. No model weights
are duplicated and no original evidence is deleted. The e5 campaign and all its
writers are independently closed before staging begins.

## Process variation

| Engine | First process corpus mean seconds | Second process corpus mean seconds |
|---|---:|---:|
{variation}

## Resources

| Worker | Samples | Peak RSS bytes | Minimum available RAM bytes | Minimum output free bytes |
|---|---:|---:|---:|---:|
{resources}

Raw observations retain root free space separately, process creation times,
thread affinities and complete process accounting. No GC, allocation, setup or
foreign-CPU observation is subtracted from public request timing.

## Every clip

{table}

## Evidence

All 320 new and 40 referenced requests pass audit, including 44 damaged-record refusals.
Every process birth is terminal before collection and reporting. Independent
Decimal arithmetic verifies {checks} values across 21 rows from integer timer ticks.
Frozen SHA256:`{audit['frozen']['sha256']}`. The collection binds {len(receipt['files'])} files.
Raw evidence is `artifacts/audio-whisper-storage-20260921`; [complete observations](observations.json)
retain unrounded values, variation and all worker results. Earlier host/revision
tables remain separate and do not establish a speedup from the storage change.
'''
    section = f'''### Audio: matched AMD Whisper baseline

AMD EPYC 9V74, logical CPU 2, .NET 10.0.8, Microsoft ORT 1.29.0, product `{audit['product_source'][:7]}`.
Twenty clips contain {float(headline['audio_seconds']):.3f} seconds of audio. Complete application calls
include frontend, inference, decoding and owned results; loading and file access
are excluded.

{short}

All 240 measured and 80 warmup calls pass application, ownership and resource checks.
Two fresh processes per engine each run one warmup and three measured passes.
The [complete report](tests/audio/whisper-amd-storage/results.md) includes every
clip, process variation and memory. The [earlier disk failure](tests/audio/whisper-amd/disk-failure-20260921.md)
remains recorded; this separate run uses verified memory-backed evidence storage.
These are descriptive results without a calibrated parity claim. Existing
encoder/logit numerical failures remain open.

'''
    benchmark = ROOT/'BENCHMARK.md'; before = benchmark.read_text(encoding='utf8')
    changed = replace_pending(before,section)
    changed = re.sub(r'^## Current results — \d{4}-\d{2}-\d{2} UTC$',
        '## Current results — '+datetime.datetime.now(datetime.timezone.utc).date().isoformat()+' UTC',changed,count=1,flags=re.M)
    write(observations,audit)
    with report.open('x',encoding='utf8') as stream:stream.write(text)
    snapshots=BASE/'document-snapshots';snapshots.mkdir()
    (snapshots/'BENCHMARK-before.md').write_bytes(benchmark.read_bytes())
    benchmark.write_text(changed,encoding='utf8',newline='\n')
    support=ROOT/'docs/model-support.md';support_before=support.read_text(encoding='utf8')
    (snapshots/'model-support-before.md').write_bytes(support.read_bytes())
    support.write_text(support_before+'\nA subsequent [matched AMD Whisper comparison](../tests/audio/whisper-amd-storage/results.md) completes all 320 fresh timing calls using verified memory-backed evidence storage. Both twenty-request conformance gates are independently rechecked and reused. The earlier disk failure and full encoder/logit numerical limits remain preserved.\n',encoding='utf8',newline='\n')
    for path in [benchmark,support]:shutil.copyfile(path,snapshots/path.name)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    files.update({p.relative_to(ROOT).as_posix():pin(p) for p in [report,observations]})
    write(BASE/'closed.json',dict(passed=True,files=files,births=receipt['births'],utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(passed=True,closure=pin(BASE/'closed.json'),headline=audit['table'][0])))


if __name__=='__main__':
    main()
