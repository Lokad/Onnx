"""Prospective exact-clock regression limits for Parakeet's shared-kernel change."""
from fractions import Fraction
from protocol import ROLES

def evaluate(table):
    assert len(table)==4 and sorted(r['audio_seconds'] for r in table)==[10,10,10,30]
    def exact(value):
        f=Fraction(**value['exact_mean']);assert f>0;return f
    controls=[];gates=[]
    for row in table:
        limit=Fraction(110 if row['audio_seconds']==30 else 120,100)
        for role in [*ROLES,'ort']:
            means=[exact(p) for p in row[role]['processes']]
            assert len(means)==2 and exact(row[role])==sum(means)/2
            ratio=max(means)/min(means)
            controls.append(dict(name=row['name'],role=role,process_ratio=float(ratio),limit=float(limit),passed=ratio<=limit))
        ratio=exact(row['candidate'])/exact(row['selected'])
        gates.append(dict(name=row['name'],control='selected',ratio=float(ratio),limit=1.05,passed=ratio<=Fraction(105,100)))
    stable=all(r['passed'] for r in controls);passed=all(r['passed'] for r in gates)
    return dict(controls_passed=stable,regression_threshold_passed=passed,admitted=stable and passed,controls=controls,gates=gates,
        policy='All clocks retained. Three engines: full dialogue process ratio<=1.10, crops<=1.20. All four candidate/selected means<=1.05; no unchanged retry.')
