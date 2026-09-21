"""Summarize all closed trials without removing failed candidates or samples."""
from common import *

def main():
    destination=TOOLS/'results-20260921.md'
    assert not destination.exists() and not (TOOLS/'observations-20260921.json').exists()
    trials=[]
    for suffix in ['', '-v2', '-v3']:
        base=ROOT/('artifacts/pyannote-portable-row-groups'+suffix+'-20260921')
        closed=read(base/'closed.json');verify(closed['files'])
        for identity in closed['identities']:terminal(identity)
        trials.append(dict(name=suffix or 'original',artifact=rel(base),closure=pin(base/'closed.json'),analysis=read(base/'analysis.json'),analysis_pin=closed['analysis']))
    broad,guarded,conditioned=[t['analysis'] for t in trials]
    assert not broad['eligible'] and not guarded['eligible']
    last=trials[-1];eligible=conditioned['eligible']
    lines=['# Portable packed row-group experiment','',
        ('The guarded composition passes the fixed kernel admission gates after complete-workload conditioning. It is eligible for a separate convolution candidate; complete public and AMD performance remain unqualified.' if eligible else
         'The conditioned guarded composition does not pass the fixed kernel admission gates. Preserve all three trials; no convolution integration or application speedup is established.'),'',
        'All three trials preserve every tested output bit against the existing two-row kernel and an independent ordered reference. Each has1,022cases /790,900output values, covering zero/nonzero accumulation, offsets, input and output guards, empty dimensions, vector and narrow tails, plus every captured tiled-convolution geometry. Product Core remains byte-identical.','',
        '| Shape rows/reduction/columns | Broad candidate / baseline | Short-reduction guard | Guard + complete conditioning |',
        '|---|---:|---:|---:|']
    for a,b,c in zip(broad['rows'],guarded['rows'],conditioned['rows']):
        assert tuple(a[k] for k in ['m','n','k'])==tuple(b[k] for k in ['m','n','k'])==tuple(c[k] for k in ['m','n','k'])
        lines.append(f"| {a['m']}/{a['n']}/{a['k']} | {a['candidate_baseline']:.6f} | {b['candidate_baseline']:.6f} | {c['candidate_baseline']:.6f} |")
    lines += ['',
        'The broad candidate fails because the two first-convolution reduction9shapes regress about21%. Its geometric mean0.841442does not override the fixed worst-shape gate. The second trial keeps the existing two-row path for reductions below64. It removes that regression, but the first shape reaching the three-row kernel,32/288/160, changes from0.926269to1.916098. Both trials pass every process-repeatability control. Neither qualifies.','',
        'The third trial tests invocation history explicitly: a complete sixteen-shape warmup precedes the original per-shape warmup and measurements. This reflects warmed application use and changes no matrix arithmetic, runtime setting, fixture or acceptance limit. It does not identify a particular JIT tier or prove the cause of the earlier history dependence. All conditioning counts and durations are retained.','',
        '## Protocol and limits','',
        'Windows i7-14700KF, CPU2, normal.NET10.0.12, no runtime overrides or forced GC. Each trial builds a standalone consumer against qualified Core0d098ba5. No product assembly is rebuilt. Six measured blocks per shape run in each of four fresh processes, baseline/candidate/candidate/baseline:384blocks per trial. Every timed iteration clears the destination, packs B and consumes the packed matrix. The deterministic iteration count targets one billion arithmetic operations per block; allocations and input generation are outside timing.','',
        'The original and guarded trials warm each shape for at least one second. The final trial additionally warms every shape for at least one second before any measurements. The controller runs onCPU0. All process wall and CPU readings, input preservation, output digests and warmup counts remain available. CPU counters have coarser resolution than the shortest blocks; they are not substituted for wall timings.','',
        'Fixed gates require both process means within1.10 for each role and shape, a candidate/baseline geometric mean at most0.95, and no shape above1.05. Numerical and resource checks must also pass. This is eligibility for further qualification, not an application speed estimate. No shape or failed sample was dropped.','',
        '| Trial | All process controls | Geometric mean | Worst shape | Kernel admission | Resource samples |',
        '|---|---|---:|---:|---|---:|']
    for trial in trials:
        a=trial['analysis']
        lines.append(f"| {trial['name']} | {a['controls_passed']} | {a['geomean_candidate_baseline']:.6f} | {a['max_shape_candidate_baseline']:.6f} | {a['eligible']} | {a['resource_samples']} |")
    lines += ['',
        'Launch availability is at least8GiB, with4GiB owned RSS,1GiB minimum available memory,20GiB disk,1GiB output and900seconds per child. All recorded identities are terminal. The guarded trial retains a premature auditor invocation that refused a still-running controller before analysis or closure output; its final unchanged auditor ran after actual termination.','',
        '## Source and qualification scope','',
        'The [source review](source-review-20260921.md) compares Lokad’s packed two/three-row kernels with the retained Microsoft ORT1.29FMA3 assembly. The candidate assigns a multiple of six rows to the existing three-row kernel and leaves zero/two/four rows to the two-row kernel. Full and final patch dimensions come from every distinct non-pointwise embedding convolution. Synthetic operands establish arithmetic and geometry, not complete-model performance.','',
        'The original public API diagnostic remains48passing calls and55–57%of selected full-request thread time in the two-row kernel. An eligible successor must preserve existing scalar/odd-shape and experimental-dispatch fallbacks, pass captured graph and public native checks, and receive fresh complete-application timing with Microsoft ORT. Accepted BENCHMARK timing tables and the frozen AMD payload remain unchanged.','',
        '## Evidence and reproduction','',
        'Use C:/Python313/python.exe -X utf8 -B from the repository root. Original commands are prepare.py/run.py/audit.py; guarded successors use _v2; conditioned successors use _v3. Every tool refuses existing outputs. Existing sources, failures and closures must not be overwritten. The report command verifies all closed file pins before writing these summaries.','']
    for trial in trials:
        c=trial['closure'];a=trial['analysis_pin']
        lines += [f"{trial['artifact']}: closure{c['bytes']}bytes, SHA256`{c['sha256']}`; analysis SHA256`{a['sha256']}`.",'']
    destination.write_text('\n'.join(lines),encoding='utf8')
    save(TOOLS/'observations-20260921.json',dict(trials=trials,scope='Retained kernel experiments only; no application or AMD performance claim.'))
    print(dict(report=rel(destination),eligible=eligible))

if __name__=='__main__':main()
