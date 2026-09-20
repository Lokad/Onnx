"""Independent Decimal arithmetic and immutable input snapshots for the review."""
from pathlib import Path
from decimal import Decimal,localcontext
import datetime,hashlib,json,re,shutil

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/e5-remaining-gap-review-v2-20260920'


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def dec(path):return json.loads(path.read_text(encoding='utf-8'),parse_float=Decimal)


def mean(values):
    values=list(values);assert values;return sum(values)/len(values)


def main():
    receipt=json.loads((BASE/'receipt.json').read_text());assert receipt['passed']
    for group in ['bindings','outputs']:
        for name,wanted in receipt[group].items():assert pin(ROOT/name)==wanted,name
    observed=dec(BASE/'observations.json');public=dec(ROOT/'artifacts/e5-public-ort-20260919/summary.json')
    resident=dec(ROOT/'artifacts/e5-interleaved-processes-v3-20260920/aa-audit.json')
    report=(ROOT/'tests/e5/remaining-gap/results-20260920.md').read_text(encoding='utf-8')
    assert len(observed['gaps'])==20;checked=0
    with localcontext() as context:
        context.prec=50
        for row in observed['gaps']:
            if row['protocol']=='isolated':
                times=lambda policy:mean(t for r in public['reports'] if r['name']==row['case'] and r['config']==policy for t in r['execute']['samples'])
                managed,native=times(row['policy']),times('ort')
            else:
                value=next(r for r in resident['timing']['cases'] if r['case']==row['case'] and r['policy']==row['policy'])
                roles=value['boundaries']['execute']['role_mean_seconds']
                managed,native=1000*mean(roles[k] for k in ['A','B','C']),1000*roles['N']
            expected=dict(managed_ms=managed,ort_ms=native,ratio=managed/native,target_ms=Decimal('1.05')*native,
                          deficit_ms=max(Decimal(0),managed-Decimal('1.05')*native))
            expected['required_reduction_percent']=100*expected['deficit_ms']/managed
            for key,value in expected.items():assert abs(value-row[key])<Decimal('1e-11'),(row['case'],key,value,row[key]);checked+=1
            cells=f"| {row['case']} | {row['policy']} | {managed:.4f} | {native:.4f} | {expected['ratio']:.4f} | {expected['deficit_ms']:.4f} | {expected['required_reduction_percent']:.3f}% |"
            assert cells in report,cells
        norms=dec(ROOT/'tests/e5/layernorm-minimum/observations-20260920.json')
        for row in observed['normalization']:
            product=mean(r['mean_ms'] for r in norms['rows'] if r['case']==row['case'] and r['variant']=='Product')
            wide=mean(r['mean_ms'] for r in norms['rows'] if r['case']==row['case'] and r['variant']=='Wide')
            assert abs(product-wide-row['reduction_ms'])<Decimal('1e-13')
            assert abs(100*(1-wide/product)-row['reduction_percent'])<Decimal('1e-12');checked+=2
        cache=dec(ROOT/'tests/e5/fingerprint-cache/observations-20260920.json')
        values={variant:mean(mean(Decimal(r['ticks'])/w['frequency']/w['repeats'] for r in w['samples'] if r['variant']==variant) for w in cache['workers']) for variant in [0,3]}
        assert abs(values[0]-values[3]-observed['fingerprint_seconds']['difference'])<Decimal('1e-17');checked+=1
    links=re.findall(r'\]\(([^)]+)\)',report)
    for link in links:assert ((ROOT/'tests/e5/remaining-gap')/link.split('#')[0]).resolve().exists(),link
    # Preserve selected inputs locally so future source edits do not erase this review's provenance.
    snapshots=BASE/'inputs';snapshots.mkdir()
    for name,wanted in receipt['bindings'].items():
        target=snapshots/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target);assert pin(target)==wanted
    value=dict(passed=True,checked_decimal_values=checked,displayed_rows=20,links=len(links),snapshots=len(receipt['bindings']),receipt=pin(BASE/'receipt.json'))
    with (BASE/'independent-verification.json').open('x') as stream:json.dump(value,stream,indent=2)
    files={p.relative_to(BASE).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    reports={p.relative_to(ROOT).as_posix():pin(p) for p in [ROOT/'tests/e5/remaining-gap/results-20260920.md',ROOT/'tests/e5/remaining-gap/observations-20260920.json',Path(__file__)]}
    with (BASE/'closed.json').open('x') as stream:json.dump(dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=files,reports=reports,scope=observed['scope']),stream,indent=2)
    print(json.dumps(dict(**value,closed=pin(BASE/'closed.json'),files=len(files))))


if __name__=='__main__':main()
