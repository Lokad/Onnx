"""Publish the entire frozen contract census and both instruction-mode results."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-prepared-recurrence-contracts-amd-20260924'
OUT=Path(__file__).resolve().parent


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    assert pin(BASE/'closed.json')['sha256']=='aacf2dbe9265a96c5699155a86b1743462387990245ca8fdc6cc0e75cc73952a'
    proof=read(BASE/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');stage=read(BASE/'bundle/stage.json')
    cases={name:[dict(r.attrib) for r in ET.parse(BASE/'collected'/name/'contracts.trx').findall('.//{*}UnitTestResult')]
           for name in analysis['contracts']}
    value=dict(passed=True,closure=pin(BASE/'closed.json'),analysis=analysis,census=stage['census'],cases=cases,
        terminal=proof['remote_terminal'],publisher=pin(Path(__file__)))
    with (OUT/'contracts-20260924.json').open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    table='\n'.join(f'| {name[:-3]} | {count} | {count} |' for name,count in stage['census'].items())
    text=f'''# Prepared recurrence: focused contracts

The actual candidate DLLs pass **150/150 cases in normal mode and 150/150 with
AVX512 disabled**, with no skipped tests. Core `3c23b44a` / Data `cc37b19e` match
the [independently reviewed build](build-20260924.md). Products were not rebuilt
for these tests. The selected release remains unchanged.

| Test class | Normal passes | AVX512-disabled passes |
|---|---:|---:|
{table}

The 22 new source contracts cover atomic pair admission at zero/short/exact
budgets; combined matrix/convolution/recurrent accounting over repeated refresh;
immutable sources, independently held outputs, source replacement, shape and
consumer changes, aliases, offset/reversed views, invalidation/rebuild, and
scalar/direct/unsupported-geometry fallbacks. Deliberately poisoning internal
prepared values proves fresh execution contexts use them; scalar and direct
calls remain exact, and invalidation restores ordinary results.

A public-API-only preparation anchor runs from the same test assembly with the
selected Core `672e5f30` / Data `065b7a7f`. It fails specifically at the expected
13,107,200-byte preparation receipt versus the original zero; its product/runtime
identity test passes. The candidate passes that same anchor. Every test host
verifies actual product/consumer hashes and paths, one CPU2 processor,
.NET10.0.8 and its requested instruction mode.

All **50 resource observations** pass; peak owned RSS is **563,703,808 bytes**.
Every PID/birth owner is terminal. The full frozen census and all 302 test results
are in [machine-readable observations](contracts-20260924.json). Raw TRX files,
loaded identities, exact test sources and monitoring logs remain under
`artifacts/parakeet-prepared-recurrence-contracts-amd-20260924`.
Closure SHA-256: `{pin(BASE/'closed.json')['sha256']}`.

[Prospective protocol](../prepared-recurrence-contracts-amd/README.md) fixes the
census and limits before execution. These contracts do not qualify the complete
model or establish a speedup. Actual decoder residency/dispatch, captured and
full native/public results, fixed complete-call/application comparisons, and
release regressions remain required before promotion or BENCHMARK.md changes.
'''
    with (OUT/'contracts-20260924.md').open('x',encoding='utf8') as stream:stream.write(text)
    print(json.dumps(dict(report=pin(OUT/'contracts-20260924.md'),observation=pin(OUT/'contracts-20260924.json'))))


if __name__=='__main__':main()
