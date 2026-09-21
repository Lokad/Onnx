"""Recompute visible counts and independent native decisions from closed raw results."""
from pathlib import Path
import json
from prepare import ROOT,BASE,PRIOR,pin,read,write
from audit import terminal


def decisions(value):
    if isinstance(value,dict):return {k:decisions(v) for k,v in value.items() if k not in ['no_speech_probability','average_log_probability']}
    if isinstance(value,list):return [decisions(v) for v in value]
    return value


def main():
    closed=read(BASE/'local-closed.json');assert closed['passed'] and closed['private_prototype'] and not closed['benchmark']
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    terminal(closed['births']);folder=BASE/'local';worker=folder/'worker';value=read(worker/'result.json')
    native=read(PRIOR/'native-corrected/manifest.json');audit=read(BASE/'local-audit.json');tracked=Path(__file__).parent
    assert read(tracked/'local-observations-20260921.json')==audit
    report=(tracked/'local-results-20260921.md').read_text(encoding='utf-8')
    for i,row in enumerate(value['cases']):
        reference=native['cases'][i%4];assert row['name']==reference['name'] and decisions(row['result'])==decisions(reference['result'])
        expected=f"| {row['name']}{' (repeat)' if row['repeat'] else ''} | {len(row['result']['windows'])} | {len(row['result']['segments'])} | {row['result']['stop_reason']} | Yes |"
        assert expected in report
    assert value['cases'][0]['result']==value['cases'][4]['result']
    assert value['short_recovery']==value['short_regression']==value['concurrent_speech'][0]['result']
    assert max(r['start'] for r in value['concurrent_speech'])<min(r['end'] for r in value['concurrent_speech'])
    assert len(value['cases'])+len(value['concurrent'])+len(value['concurrent_speech'])+4==13 and value['refusals']==16
    samples=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
    peak=max(sum(m['rss'] for m in s['members']) for s in samples);available=min(s['available'] for s in samples)
    assert audit['resources']['samples']==len(samples) and audit['resources']['peak_rss']==peak and audit['resources']['min_available']==available
    assert f'**{len(samples):,}** resource samples' in report and f'**{peak:,} bytes**' in report and f'**{available:,} bytes**' in report
    result=dict(passed=True,closure=pin(BASE/'local-closed.json'),pins=len(closed['files']),native_recording_decisions=5,displayed_rows=5,
        completed_requests=13,refusals=16,overlap_verified=True,resources_recomputed=True,births=closed['births'])
    write(BASE/'local-final-verification.json',result);print(json.dumps(result))


if __name__=='__main__':main()
