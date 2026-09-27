"""Publish the complete fixed-shape verdict without editing its clocks or gates."""
import json
from pathlib import Path
from run import BASE,ROOT,pin,read,write,prepared


def main():
    prepared();closed=read(BASE/'closed.json')
    assert closed['completed'] and not closed['component_admitted']
    for name,wanted in closed['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');assert closed['analysis']==pin(BASE/'analysis.json')
    performance=analysis['performance']
    assert len(performance['controls'])==246 and sum(r['passed'] for r in performance['controls'])==208
    assert len(performance['gates'])==43 and sum(r['passed'] for r in performance['gates'])==42
    target=ROOT/'tests/parakeet/decoder-lstm-layout-profile-results/pointwise-tail-timing-observations-20260927.json'
    value=dict(closed=pin(BASE/'closed.json'),analysis=pin(BASE/'analysis.json'),identities=analysis['identities'],boundary=analysis['boundary'],
        component_admitted=False,all_outputs_exact=analysis['all_outputs_exact'],clocks=analysis['clocks'],performance=performance,resources=analysis['resources'],
        no_application_speedup_claim=True,release_admitted=False,publisher=pin(Path(__file__)))
    write(target,value);print(json.dumps(dict(published=pin(target),component_admitted=False)))


if __name__=='__main__':main()
