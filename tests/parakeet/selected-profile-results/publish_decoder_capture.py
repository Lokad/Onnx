"""Publish the independently closed actual decoder capture without duplicating tensors."""
import collections
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-decoder-lstm-capture-amd-20260924'
OUT=Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    assert pin(BASE/'closed.json')['sha256']=='28c7afe448ed16e3bb19d29232c3f90eb2d72c096196ae27792f5261afa5b64f'
    closed=read(BASE/'closed.json');assert closed['passed']
    for name,wanted in closed['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');native=read(BASE/'collected/native/result.json')
    capture=read(BASE/'collected/capture/output/result.json');spec=read(BASE/'collected/capture-spec.json')
    summaries=[]
    for case in spec['cases']:
        rows=[r for r in native['rows'] if r['name']==case['name']]
        summaries.append(dict(name=case['name'],steps=len(case['steps']),lstm_calls=2*len(case['steps']),
            decoder_controls=4*len(case['steps']),native_arrays=len(rows),max_error=max((r['max_error'] for r in rows),default=0)))
    observation=dict(passed=True,closure=pin(BASE/'closed.json'),analysis=analysis,cases=summaries,
        nodes=spec['nodes'],original_node_count=capture['original_node_count'],initializer_count=capture['initializer_count'],
        native_libraries=native['libraries'],terminal=closed['remote_terminal'],publisher=pin(Path(__file__)))
    output=OUT/'decoder-lstm-capture-20260924.json'
    with output.open('x',encoding='utf8') as stream:json.dump(observation,stream,indent=2,allow_nan=False);stream.write('\n')
    table='\n'.join(f"| {r['name']} | {r['steps']} | {r['lstm_calls']} | {r['native_arrays']} | {r['max_error']:.9g} |" for r in summaries)
    text=f'''# Actual Parakeet decoder LSTM capture

Both recurrent nodes are now captured and independently qualified against
Microsoft ORT on every one of the selected trajectory's 190 decoder steps.
The selected Core `672e5f30` and Data `065b7a7f` remain unchanged. This is a
correctness fixture bank for the prepared-recurrence trial, with no speed claim.

| Case | Decoder steps | Complete LSTM calls | Native output arrays, two repeats | Largest scaled error |
|---|---:|---:|---:|---:|
{table}

Each nonempty case ran twice normally and twice with additional per-execution
output bindings. All **760 original decoder executions / 3,040 output arrays**
matched the prior selected results exactly, including token and duration
decisions. Each pass carried its own accepted recurrent states. All original
optimized node fields/attributes, output names, initializer references/shapes/
contents and held outputs remained unchanged. No original model was edited.

Two tiny native graphs copy the original LSTM nodes byte for byte, preserve
opsets/IR, and feed captured weights and states at the actual one-step, batch-one,
input/hidden-size-640 geometry. ORT **1.29.0 CPUExecutionProvider** runs with one
intra/inter thread, sequential execution, all optimizations and no spinning.
All **2,280 native arrays / 1,459,200 values** pass
`abs(managed-native)/max(1,abs(native)) <= 1e-4`. The largest error is
**{analysis['native']['max_error']:.12g}**. Native repeats are exact; inputs and
held outputs remain unchanged. Fed-weight graphs are correctness instruments;
they do not establish an ORT performance baseline.

Capture tensors occupy **27,374,080 bytes in 443 unique files**, below the frozen
64 MiB cap. Every W/R/B constant matches the original model's raw bytes. No encoder
execution, model download or full model copy was needed. All 115 resource samples
pass; peak owned RSS is **754,020,352 bytes**. Every worker is terminal.

[Prospective protocol and commands](../decoder-lstm-capture-amd/README.md) fix
all cases, resource bounds and correctness controls before execution.
[Machine-readable observations](decoder-lstm-capture-20260924.json) retain case
counts, both node descriptors, constant hashes, loaded native library identities
and resource summaries. All raw captured/native arrays, control rows and logs
remain under `artifacts/parakeet-decoder-lstm-capture-amd-20260924`.
Closure SHA-256: `{pin(BASE/'closed.json')['sha256']}`.

Next: implement the isolated cache within the existing aggregate 64 MiB decoder
budget, preserve the ordered projection arithmetic, prove lifecycle/dispatch,
then qualify complete trajectories and fixed performance comparisons. The
release and BENCHMARK.md change only after application and regression admission.
'''
    with (OUT/'decoder-lstm-capture-20260924.md').open('x',encoding='utf8') as stream:stream.write(text)
    print(json.dumps(dict(observation=pin(output),report=pin(OUT/'decoder-lstm-capture-20260924.md'))))


if __name__=='__main__':main()
