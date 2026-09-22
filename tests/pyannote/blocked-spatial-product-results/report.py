"""Publish closed product qualification without implying application speed."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def read(p):return json.loads(p.read_text())


def main():
    folders={
        'raw_local':ROOT/'artifacts/pyannote-blocked-spatial-raw-graphs-20260922',
        'amd':ROOT/'artifacts/pyannote-blocked-spatial-product-amd-v3-20260922',
        'parakeet':ROOT/'artifacts/pyannote-blocked-spatial-parakeet-20260922'}
    closures={}
    for name,folder in folders.items():
        proof=read(folder/'closed.json');assert proof['passed']
        for rel,wanted in proof['files'].items():assert pin((ROOT if name=='parakeet' else folder)/rel)==wanted,rel
        closures[name]=pin(folder/'closed.json')
    amd=read(folders['amd']/'analysis.json');para=read(folders['parakeet']/'analysis.json')
    failures={}
    for name in ['blocked-spatial-product-amd','blocked-spatial-product-amd-v2']:
        folder=ROOT/('artifacts/pyannote-'+name+'-20260922');proof=read(folder/'failure-closed.json')
        assert proof['retained_failure'] and not proof['passed']
        failures[name]=dict(closure=pin(folder/'failure-closed.json'),analysis=read(folder/'failure-analysis.json'))
    observations=dict(closures=closures,failures=failures,amd=amd,parakeet=para,
        raw_local=read(folders['raw_local']/'output/256.json'),
        amd_results={mode:read(folders['amd']/'collected'/mode/'result.json') for mode in ['raw-256','raw-512','layers-256','layers-512']})
    path=OUT/'observations-20260922.json';assert not path.exists()
    path.write_text(json.dumps(observations,separators=(',',':'),allow_nan=False)+'\n',encoding='utf8')
    summary=f'''# Prepared convolution: actual product and Parakeet qualification

The unselected prepared-convolution product passes its complete raw/ordinary-graph
qualification locally and on AMD in both AVX2 and AVX512 modes. Complete Parakeet
regression also passes against the selected runtime. **No new application timing
or production selection is claimed.** Pyannote remains first, Parakeet second,
and Whisper deferred.

| Qualification, per runtime/mode | Result |
|---|---|
| Raw helper arithmetic | 2,648 cases, 20 supplemental cases, 10 invalid/alias guards pass |
| Ordinary raw graphs | 2,668 controls and 5,336 prepared requests pass |
| Finite / nonfinite fallback graphs | 2,504 / 164; no changed output bits or NaN payloads |
| Captured Pyannote layers, each AMD width | 108 graphs, 216 requests, 119,823,360 values; exact selected outputs |
| Captured layer native tolerance | Maximum scaled error 3.814697265625e-6; zero failures |
| Parakeet native regression | All 784 arrays / 3,090,494 values identical to selected |
| Public Parakeet transcription | All twenty clips identical; inputs and held outputs unchanged |

The raw cases exercise the actual internal product implementation through a
friend test consumer and normal graph calls. Prepared-weight residency, requested
scratch, fallback dispatch, repeated requests, and ownership are checked. The
captured layers cover three crops and both instruction widths. The AMD run uses
.NET10.0.8 on EPYC9V74, CPU2; monitoring uses CPU0. All100target resource samples
pass, with peak RSS2,635,485,184bytes. Its two local numerical workers also pass
all52resource samples. Product Core is `3c2f16b0` / Data `6318cf48` throughout.

Parakeet's selected control is Core `1279b4b6` / Data `4e602d9f`. The unchanged
native/public consumers run four sequential Windows .NET10.0.12 workers. All855
resource samples pass, with peak RSS8,571,314,176bytes. Both native workers
reproduce exactly the three existing Windows failures at decoder step26/outputs:
english-16k, english-frame-limit, and english-repeat. Their maxima are unchanged.
This is a passing regression with preserved native failures, **not a full native
numerical pass**. No worker elapsed time is used as a performance comparison.

The first AMD attempt stopped before cases because `DOTNET_EnableAVX512F=0` did
not select eight lanes. The corrected setting is `DOTNET_EnableAVX512=0`, as
defined in the [.NET10.0.8 source](https://github.com/dotnet/runtime/blob/v10.0.8/src/coreclr/inc/clrconfigvalues.h#L615).
The next preparation built both consumers but its verifier rejected the renamed
compiler cache field associated with that string. Both failures remain recorded.
The successful successor reuses those binaries and explicitly checks exactly
two literal operands and the corresponding cache-field load/store; every other
method instruction and public declaration remains identical. Product code did
not change. All local and remote owners are terminal.

Reproduction tools: [raw graph qualification](../blocked-spatial-raw-graphs/README.md),
[corrected AMD qualification](../blocked-spatial-product-amd-v3/README.md), and
[Parakeet regression](../blocked-spatial-parakeet/README.md).
The [prospective application campaign](../blocked-spatial-app-amd/README.md)
retains the original full-model, meeting, repeatability, and speed gates before
any root integration. The selected complete-application results remain Pyannote
15.361590s versus ORT9.094606s (1.689088), and Parakeet75.134536s versus
ORT39.464229s (1.903864); those are separate retained campaigns.

The [complete observations](observations-20260922.json) retain all raw graph and
AMD layer rows, Parakeet array comparisons/native failures, resource summaries,
compiled consumer changes, and both failed attempts. File identity:
`{pin(path)['sha256']}` ({pin(path)['bytes']:,}bytes).
Closures: localraw `{closures['raw_local']['sha256']}`;
AMD `{closures['amd']['sha256']}`;
Parakeet `{closures['parakeet']['sha256']}`.
'''
    (OUT/'results-20260922.md').write_text(summary,encoding='utf8')
    print(json.dumps(dict(observations=pin(path),report=pin(OUT/'results-20260922.md'))))


if __name__=='__main__':main()
