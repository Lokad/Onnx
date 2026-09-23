"""Publish actual-product scope and exact numerics, reusing identical canonical rows."""
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BUILD=ROOT/'artifacts/pyannote-winograd-register-transform-build-amd-20260923'
NUM=ROOT/'artifacts/pyannote-winograd-register-transform-numerics-amd-20260923'
ROWS=ROOT/'tests/pyannote/winograd-output-blocks-results'

def read(p):return json.loads(p.read_text(encoding='utf8'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def save(p,value):
    assert not p.exists();p.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')

def main():
    for folder,digest in [(BUILD,'0da29afbb3f75e92aeb66cc5ffc2667984f24f034d6aa4275b7e3bd31d21bbec'),(NUM,'431fffeebbbcfd8ee751fd1cf343db251a7c08e4dca79bcd8b55c55628187161')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    analysis=read(NUM/'analysis.json');build=read(BUILD/'analysis.json');assert analysis['numerically_admitted']
    runs={}
    for name in analysis['results']:
        value=read(NUM/'collected'/name/'result.json');canonical=ROWS/(value['mode']+'-rows-20260923.json')
        assert value['rows']==read(canonical)
        runs[name]=dict(result=pin(NUM/'collected'/name/'result.json'),metadata={k:v for k,v in value.items() if k!='rows'},canonical_rows=canonical.relative_to(ROOT).as_posix(),canonical_pin=pin(canonical))
    save(OUT/'numerics-observations-20260923.json',dict(closure=pin(NUM/'closed.json'),analysis=analysis,runs=runs))
    save(OUT/'build-observations-20260923.json',dict(closure=pin(BUILD/'closed.json'),analysis=build))
    text=f'''# Register-scheduled Winograd input transform: numerical qualification

**Passed; no performance result yet.** The actual normally built candidate
Core `eaea0aea` changes only `TransformWinogradInputContiguous`. All 3,178 other
Core methods, 697 Data methods and public declarations match the selected
product. All 420 root source files remain unchanged.

One unchanged numerical consumer binds the actual current Core `521bae17` and
candidate DLL through typed delegates. No kernel implementation is compiled
into the consumer. Both products and both instruction widths return the same
full result rows as the closed current-product references, including all
expanded synthetic cases. The compiler may schedule loads differently; the
arithmetic, admission guards, scratch layout and public behavior remain exact.

| Coverage per product and width | Cases | Output values | Maximum Winograd scaled error |
|---|---:|---:|---:|
| Synthetic independent double reference | 3,456 | 45,958,656 | 0 |
| Captured native reference | 87 | 98,734,080 | 1.8477439880371094e-6 |
| Independent range checks | 10,566 | — | Exact Boolean agreement |

All 21 numerical/budget refusals, 24 alias/extent cases and two m16 geometry
refusals pass per product and width. Held/repeated output, read-only operand
and buffer sentinel checks pass. The native scaled-error bound remains 1e-4.
Zero synthetic error applies to these dyadic patterns. The unchanged direct
control reaches at most 3.814697265625e-6 on captured native references.

All eleven build/numerical workers are terminal with exit zero. All
{analysis['resources']} resource observations pass; peak owned RSS is
{analysis['peak_rss']:,} bytes. SDK 10.0.204/runtime 10.0.8, CPU2 workers and
CPU0 supervisor. Normal execution and AVX512-disabled execution are both checked.
The 21 audit tests reject missing/duplicate cases, output/statistic drift,
invalid error claims and incorrect process/product identities.

The [structured numerical evidence](numerics-observations-20260923.json) records
each original result digest, all role identities and the complete audit. The
publisher verifies that every fresh row equals the already published
[synthetic rows](../winograd-output-blocks-results/raw-rows-20260923.json) and
[captured rows](../winograd-output-blocks-results/captured-rows-20260923.json).
Those identical canonical rows are shared; all eight original worker results
remain in the closed artifact.

The [normal build evidence](build-observations-20260923.json) retains the complete
single-method scope and all 74 resource observations, peak 502,759,424 bytes.
No method is added, removed or renamed. The selected root product is unchanged.

Numerical closure: `{pin(NUM/'closed.json')['sha256']}`.
Build closure: `{pin(BUILD/'closed.json')['sha256']}`.
Consumer: `{analysis['consumer']['sha256']}`.

The [source proposal](../winograd-register-transform-source/README.md) and
[numerical protocol](../winograd-register-transform-prototype/README.md) specify
the exact scope. All-tier generated-code review must pass before a separate
complete-call screen. BENCHMARK.md's current release comparisons are unchanged.
'''
    target=OUT/'numerics-20260923.md';assert not target.exists();target.write_text(text,encoding='utf8')
    print(json.dumps(dict(report=pin(target),consumer=analysis['consumer'])))

if __name__=='__main__':main()
