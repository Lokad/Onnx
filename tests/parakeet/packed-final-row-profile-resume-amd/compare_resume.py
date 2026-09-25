"""Run the unchanged graph-edge comparison against the audited resumed capture."""
import json
from checks import ORIGINAL,module
from resume import BASE,pin,read,write,prepared


def main():
    prepared();proof=read(BASE/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    assert not (BASE/'comparison-driver.json').exists()
    compare=module('retained_projection_comparison',ORIGINAL/'compare.py')
    write(BASE/'comparison-driver.json',dict(passed=True,original=pin(ORIGINAL/'compare.py'),
        closure=pin(BASE/'closed.json'),only_override='BASE: audited resume collection; original comparison function unchanged'))
    compare.BASE=BASE
    compare.main()


if __name__=='__main__':main()
