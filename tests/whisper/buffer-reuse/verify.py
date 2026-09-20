"""Recompute closure, source provenance, raw allocation totals and displayed cells."""
from decimal import Decimal
from pathlib import Path
import hashlib,json,re,subprocess,tarfile
from deploy import BASE,ROOT
from close import terminal
from protocol import pin,read,write


def main():
    closed=read(BASE/'closed.json');assert closed['passed'] and closed['prototype_only'] and not closed['benchmark']
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    terminal(closed['births']);base=BASE/'collected';state=read(base/'campaign/identity.json')
    manifest=read(base/'manifests/whisper.json');expected={c['name']:c for c in manifest['cases']}
    audit=read(BASE/'audit.json');folder=Path(__file__).parent;assert read(folder/'observations-20260920.json')==audit
    report=(folder/'results-20260920.md').read_text(encoding='utf-8');display=[];calls=0
    for run,observation in zip(state['runs'],audit['observations'],strict=True):
        worker=base/run['output']/'worker';paths=sorted(worker.glob('[0-9][0-9][0-9].json'));assert len(paths)==observation['calls']
        values=[json.loads(p.read_text(),parse_float=Decimal) for p in paths]
        for i,row in enumerate(values):
            case=manifest['cases'][i%20];assert row['name']==case['name'] and row['pass']==i//20
            assert row['result']==case['expected'] and row['input_sha256']==case['raw_sha256'] and row['ownership'] is True
            assert row['pools']['encodingExecution']['cache_budget']==536870912
            assert row['pools']['encodingExecution']['cache_bytes']<=536870912
            if i:assert row['pools']['encodingExecution']['allocated_new_bytes']<=16777216
            calls+=1
        allocations=[v['allocated_bytes'] for v in values];assert sum(allocations)==observation['allocated_total']
        samples=[json.loads(s) for s in (base/run['output']/'samples.jsonl').read_text().splitlines()]
        peak=max(sum(m['rss'] for m in s['members']) for s in samples);available=min(s['available'] for s in samples)
        assert peak==observation['resource']['peak_rss'] and available==observation['resource']['min_available']
        assert peak<15032385536 and available>=1073741824
        display.append(f"| {'Conformance' if run['phase']=='conformance' else 'Endurance'} | {len(values)} | {peak:,} | {available:,} | {min(allocations):,}–{max(allocations):,} |")
        if run['phase']=='conformance':
            original=[read(base/'original-prefix'/f'{i:03}.json') for i in range(16)]
            old=sum(v['allocated_bytes'] for v in original[1:]);new=sum(v['allocated_bytes'] for v in values[1:16])
            assert new*2<=old and old==audit['gate']['original_allocated_bytes'] and new==audit['gate']['prototype_allocated_bytes']
            assert f'**{old:,} to {new:,} bytes**' in report
            assert f'({Decimal(new)/Decimal(old):.3%} of the saved original prefix)' in report
    assert calls==100 and all(row in report for row in display)
    source=read(BASE/'source.json');cli=read(BASE/'cli-source.json')
    tree=subprocess.check_output(['git','ls-tree','-r',source['revision']],cwd=ROOT,text=True)
    git_blobs={line.split('\t',1)[1]:line.split()[2] for line in tree.splitlines()}
    verified=0;normalized=0
    for filename,receipt in [('original-source.tar',source),('cli-source.tar',cli)]:
        assert pin(BASE/filename)==receipt['archive']
        with tarfile.open(BASE/filename) as archive:
            for member in archive.getmembers():
                if not member.isfile():continue
                content=archive.extractfile(member).read()
                blob=lambda b:hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
                if blob(content)!=git_blobs[member.name]:
                    # Windows git archive applies core.autocrlf to text. Retain exact
                    # archive SHA256 separately; allow only CRLF-to-LF text conversion.
                    content.decode('utf-8');assert b'\0' not in content
                    assert blob(content.replace(b'\r\n',b'\n'))==git_blobs[member.name],member.name
                    normalized+=1
                verified+=1
    for receipt in [source,cli]:
        for name,wanted in receipt['files'].items():assert pin(BASE/'source'/name)==wanted,name
    for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']:
        assert pin(BASE/'source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0'/name)==pin(BASE/'bin'/name)==pin(base/'bin'/name)
    ort=read(folder/'ort-source-20260920.json')
    for name,wanted in ort['files'].items():assert pin(ROOT/'artifacts/ort-audio-memory-source-20260920'/name)=={k:wanted[k] for k in ['bytes','sha256']}
    result=dict(passed=True,closure=pin(BASE/'closed.json'),pins=len(closed['files']),git_blobs=verified,git_crlf_text_normalizations=normalized,calls=calls,displayed_rows=len(display),ort_source_files=len(ort['files']),births=closed['births'])
    write(BASE/'final-verification.json',result);print(json.dumps(result))


if __name__=='__main__':main()
