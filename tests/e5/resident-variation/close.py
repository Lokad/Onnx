"""Record the verified resident-process diagnosis without changing old verdicts."""
from pathlib import Path
import datetime,json,shutil
from analyze import ROOT,SOURCE,BASE,pin,read,write

def main():
    value=read(BASE/'analysis.json');verification=read(BASE/'verification.json');assert verification['passed'] and verification['analysis']==pin(BASE/'analysis.json')
    for name,wanted in value['inputs'].items():assert pin(ROOT/name)==wanted,name
    folder=Path(__file__).parent
    totals={}
    for boundary in ['execute','request']:
        rows=[r for r in value['summary'] if r['boundary']==boundary]
        totals[boundary]={k:sum(r[k] for r in rows) for k in ['failed_pairs','failed_pairs_same_sign','failed_in_both_halves']}
    assert totals['execute']==dict(failed_pairs=40,failed_pairs_same_sign=40,failed_in_both_halves=31)
    assert totals['request']==dict(failed_pairs=40,failed_pairs_same_sign=40,failed_in_both_halves=32)
    def line(row):
        low,high=row['process_share_range']
        return f"| {row['case']} | {row['policy']} | {row['boundary']} | {row['failed_pairs']} / 12 | {row['failed_pairs_same_sign']} | {row['failed_in_both_halves']} | {row['process_share_median']*100:.2f}% | {low*100:.2f}–{high*100:.2f}% |"
    table='\n'.join(line(row) for row in value['summary'])
    text=f'''# Persistent differences in the retained e5 resident-process run

All **40 Execute control comparisons** that failed the original per-visit1%
limit retain the same direction in both balanced halves of the run. **31** exceed
that limit in both halves. The enclosing request boundary also has40failures
with consistent direction,32of them failing in both halves. This finding supports
treating fresh-process repetition as necessary for another comparison; simply
collecting more calls in these same processes does not resolve the observed
persistent differences.

The [original experiment](../interleaved-processes/aa-results-20260920.md) remains
failed under its complete timing screen. This is a post-hoc description of its
forty cohorts and all **89,088 measured calls**, with no candidate phase, changed
gate, removed tail, corrected timing, speedup or confidence claim. The earlier
[sequential-process analysis](../deployment-variance/results-20260920.md) covers
different data and is not combined with these means.

Each cohort has three identical managed processes and one ORT process. Processes
alternate through48cycles; each24cycle half contains every role order once. The
table below counts the original three managed pairs in each of four visits.
Thus each row covers twelve pair comparisons, which are dependent comparisons,
not twelve independent experiments. `Same direction` refers to both halves of an
originally failed pair; `Both halves fail` also requires the original1%bound.

| Case | Policy | Boundary | Original failed pairs | Same direction | Both halves fail | Median process share | Process-share range |
|---|---|---|---:|---:|---:|---:|---:|
{table}

Process share is a descriptive fraction of squared deviations in a3by48table
of batch times. Each table decomposes into a persistent process mean, a common
cycle mean and the remaining process-by-cycle component. Exact integer arithmetic
proves that those three terms sum to the total for all80cohort/boundary tables.
The percentages are medians/ranges across four cohorts, not shares of total
latency, estimates of independent variance, or costs that can be removed.

For example, the first512-token Memory cohort has a B/A Execute ratio1.063928.
Its balanced halves are1.062918and1.064951; all four quarters remain above1.057.
The first8-token Default cohort has C/A1.040809, with halves1.042886and1.038667.
These persistent contrasts coexist with common cycle movement and within-process
variation. Neither this decomposition nor the retained GC/allocation counts
identifies a JIT, collector, memory-layout, frequency or hypervisor cause.

## Consequence for the next measurement

Do not retry the old interleaved or sequential protocol unchanged. Another
candidate comparison needs independent fresh-process replication and an estimator
whose uncertainty is based on that replication, with balanced/randomized order
declared before new measurements. Per-call precision cannot substitute for
process replication. The new design must retain normal runtime operation, all
primary lengths and512regression coverage, complete output/resource checks and
every sample. Matching A/A evidence must support its interpretation before a
candidate can justify changing defaults. This report does not itself qualify
such a design or establish the unchanged1.05e5 target.

The general distinction between repeated calls and higher-level executions is
also discussed by [Kalibera and Jones, Rigorous Benchmarking in Reasonable Time](https://kar.kent.ac.uk/33611/).
Their method is background for the next design; this descriptive decomposition
does not import its statistical assumptions or confidence formulas.

## Scope and reproducibility

The analysis accounts for66,816managed and22,272native measured calls,
227,583conditioning calls and5,120solo calls. Native calls are counted and their
source files verified, but are not members of the explicitly managed3process
decomposition. Conditioning and solo calls remain outside the measured boundary.
Full GC,allocation,pair,half,quarter,batch and squared-deviation records are kept
in artifacts/e5-resident-variation-20260921/analysis.json.

The original closure SHA256 is
`45fcf0832ef4f48b7f14c29fc5ea08b384b113122606dc18560e56bd5de73a17`.
All selected raw files are checked against that immutable inventory. Independent
rational row/column means verify320component values and1,440half/quarter ratios;
twenty summary rows are recomputed. Mathematical fixtures cover constant,pure
process,pure cycle,interaction,mixed large integers and malformed tables.
This analysis creates no VM load and reruns no model inference.

Run `C:/Python313/python.exe -X utf8 -B tests/e5/resident-variation/analyze.py`
only for a new artifact, followed by `verify.py` and `close.py`. Completed
create-new writers must not be rerun. All original timing failures and production
defaults remain unchanged.
'''
    # Keep prose readable while retaining all literal numerical values.
    for a,b in {'per-visit1%':'per-visit 1%','has40failures':'has 40 failures','32of':'32 of','through48cycles':'through 48 cycles','each24cycle':'each 24-cycle','original1%bound':'original 1% bound','a3by48table':'a 3-by-48 table','all80cohort/boundary':'all 80 cohort/boundary','first512-token':'first 512-token','ratio1.063928':'ratio 1.063928','are1.062918and1.064951':'are 1.062918 and 1.064951','above1.057':'above 1.057','first8-token':'first 8-token','C/A1.040809':'C/A 1.040809','halves1.042886and1.038667':'halves 1.042886 and 1.038667','and512regression':'and 512 regression','unchanged1.05e5':'unchanged 1.05 e5','for66,816managed and22,272native':'for 66,816 managed and 22,272 native','227,583conditioning':'227,583 conditioning','and5,120solo':'and 5,120 solo','managed3process':'managed three-process','verify320component':'verify 320 component','and1,440half/quarter':'and 1,440 half/quarter'}.items():text=text.replace(a,b)
    report=folder/'results-20260921.md'
    with report.open('x',encoding='utf-8') as f:f.write(text)
    write(folder/'observations-20260921.json',dict(counts=value['counts'],summary=value['summary'],totals=totals,analysis=pin(BASE/'analysis.json'),verification=verification))
    snapshots=BASE/'tools';snapshots.mkdir()
    for p in folder.glob('*.py'):shutil.copyfile(p,snapshots/p.name)
    files={str(p.relative_to(ROOT)):pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    files.update({str(p.relative_to(ROOT)):pin(p) for p in [report,folder/'observations-20260921.json']})
    write(BASE/'closed.json',dict(passed=True,diagnostic_only=True,original_timing_passed=False,files=files,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    print(json.dumps(dict(passed=True,closure=pin(BASE/'closed.json'),totals=totals)))

if __name__=='__main__':main()
