"""Publish complete-body diagnostic facts without scoring any retained clocks."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-dense-scalar-where-control-codegen-amd-20260924'
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('native_parser',ROOT/'tests/parakeet/wide-entry-first-use-results/inspect_codegen.py')
parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
pin=parser.pin


def main():
    target=HERE/'control-codegen-20260924.md';assert not target.exists()
    review=json.loads((HERE/'control-codegen-review-20260924.json').read_text())
    assert review['passed'] and review['diagnostic_only'] and not review['performance_admitted']
    assert review['closure']==pin(BASE/'closed.json')
    closure=json.loads((BASE/'closed.json').read_text())
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    for body in review['bodies']:assert pin(ROOT/body['file'])==body['identity']
    analysis=json.loads((BASE/'analysis.json').read_text())
    state=json.loads((BASE/'collected/identity.json').read_text())
    order=list(review['important']);assert len(order)==4
    lines=['# Exact-consumer Where code-generation diagnostic','',
        '**Diagnostic passed; no performance admission.** The four fresh processes reuse the rejected control’s exact consumer and identical selected release binaries. The only extra managed setting is the declared disassembly flag. No build or product change occurred.','',
        'Whole-method optimized code sizes in process order:','',
        '| Method | Outer 0 bytes | Middle 1 bytes | Middle 2 bytes | Outer 3 bytes |',
        '|---|---:|---:|---:|---:|']
    for label,title in [('provider','Public provider Where'),('float_where','Generic float Where'),('float_phase','Consumer Work<float>.Phase')]:
        rows=[[r for r in review['important'][p][label] if r['tier']=='Tier1'] for p in order]
        assert all(len(v)==1 for v in rows)
        lines.append('| '+title+' | '+' | '.join(str(v[0]['bytes']) for v in rows)+' |')
    lines+=['','The generic float entries all retain 23 profiled inlinees, no external BroadcastShape calls and four integer divisions. Their branch layouts differ, including placement of the dimension-zero branch and selection-loop blocks. Equal inline counts do not imply equal generated code. The provider retains calls for all nine supported element types. The float consumer retains the public provider call inside its timer; it does not invoke the candidate helper.','',
        'The emitted whole-method Tier1 census is:','',
        '| Element type | Generic Where sizes, processes 0 / 1 / 2 / 3 | Consumer Phase whole Tier1 |',
        '|---|---|---|']
    for dtype in ['float','long','bool','byte','int','uint','ulong','double','System.Half']:
        wheres=[];phases=[]
        for process in order:
            rows=[b for b in review['bodies'] if b['process']==process and b['tier']=='Tier1']
            wheres.append(','.join(str(b['bytes']) for b in rows if f'Tensor`1[{dtype}]:Where(' in b['method']) or 'absent')
            phases.append(','.join(str(b['bytes']) for b in rows if f'Work`1[{dtype}]:Phase(' in b['method']) or 'absent')
        lines.append('| '+dtype+' | '+' / '.join(wheres)+' | '+' / '.join(phases)+' |')
    for process in order:
        whole=[b for b in review['bodies'] if b['process']==process and b['tier']=='Tier1']
        f=next(b for b in whole if 'Tensor`1[float]:Where(' in b['method'])
        assert f['divisions']==4 and not any(':BroadcastShape(' in c for c in f['calls'])
        assert any('23 inlinees with PGO data' in p for p in f['profile'])
        provider=next(b for b in whole if 'CPUExecutionProvider:Where(' in b['method'])
        for dtype in ['bool','byte','int','long','uint','ulong','float','double','System.Half']:
            assert any(f'Tensor`1[{dtype}]:Where(' in c for c in provider['calls'])
        phase=next(b for b in whole if 'Work`1[float]:Phase(' in b['method'])
        assert any('CPUExecutionProvider:Where(' in c for c in phase['calls'])
        assert not any('DenseScalarWhere' in c for c in phase['calls'])
    lines+=['',
        '“Absent” means no such complete body appears in these retained listings; it does not mean the calls were skipped. Every case completed. The other consumer specializations emit instrumented Tier0 and Tier1-OSR bodies. A phase can execute many internal iterations while its method is entered only twice per case. Here 210 cases are float, three are Int64, and each of the other seven types has one case. Increasing the internal sample count does not increase the number of phase-method entries.','',
        'On-stack replacement (OSR) enters optimized code during an already-running invocation. A whole-method Tier1 entry is a separate compilation. The [.NET runtime explanation](https://github.com/dotnet/runtime/blob/v10.0.8/docs/design/features/OsrDetailsAndDebugging.md) describes that distinction and the effect of internal loops on benchmarking. The versioned document is retained with its hash under `artifacts/parakeet-where-control-runtime-source-20260924`.','',
        'These observations motivate a distinct consumer with one shared, non-generic batch-timing method called for every sample and round-robin warmup across all 220 cases. That is a hypothesis to test, not a proven fix. Keep the original 600 warmup / 180 measured samples, complete public calls, deterministic batches, all cases and unchanged acceptance limits. Qualify its generated code and require a new identical-binary control before any candidate race.','',
        'The diagnostics do not establish the exact tiers of the original untraced control, attribute its false 37.6% improvement to a particular mechanism, or admit M59 performance. Branch layout, the consumer and lower-level callees remain distinct possible influences. Do not repeat the unchanged rejected control, trim clocks, drop cases or relax gates.','',
        f'All five jobs pass: {analysis["resources"]:,} resource observations, {analysis["setup_count"]:,} setups, {analysis["sample_clocks"]:,} retained clocks and {analysis["public_calls"]:,} complete public calls ({analysis["measured_public_calls"]:,} within measured intervals). Peak process-tree RSS is {analysis["peak_rss"]:,} bytes. All 880 setups match the prior control’s input hashes, shapes and qualified output hashes. Exact result bits, metadata, unchanged stores and held/output ownership pass.','',
        f'All {len(review["bodies"])} emitted bodies are retained to their terminal byte count, with every named branch target resolved. The [complete body census, call lists and comparison identities](control-codegen-review-20260924.json) identify the full listings and diffs. Normalization removes only explicit relocation fields; unmarked pointer-like constants, registers, instructions and branches remain. This avoids hiding code differences, but means address changes can still appear in the diffs.','',
        f'Closure: `{review["closure"]["sha256"]}`. Tools frozen at `09b16e50`. Supervisor {state["supervisor"]["pid"]} / birth {state["supervisor"]["birth"]} and all descendants are terminal. Consumer `19f25657` (52,736 bytes), Core `672e5f30`, Data `065b7a7f`. Full collection: `artifacts/parakeet-dense-scalar-where-control-codegen-amd-20260924`.','',
        'The selected release and BENCHMARK.md remain unchanged: Parakeet / ORT 1.864, Pyannote / ORT 1.154. M59 is numerically qualified and still unmeasured.','']
    target.write_text('\n'.join(lines))
    print(json.dumps(dict(report=pin(target),review=pin(HERE/'control-codegen-review-20260924.json'),bodies=len(review['bodies']))))


if __name__=='__main__':main()
