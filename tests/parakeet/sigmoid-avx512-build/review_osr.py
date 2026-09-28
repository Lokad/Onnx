"""Read the captured optimized OSR listing without another build or test run."""
from pathlib import Path
from run import BASE, TOOLS, pin, read, write


def main():
    original = TOOLS/'review.py'
    assert pin(original) == read(BASE/'prepared.json')['tools']['review.py']
    failure = read(BASE/'capture-review-failed.json')
    assert not failure['passed'] and failure['code'] == 1
    assert failure['reviewer'] == pin(original)
    source = original.read_text()
    old = "optimized=[b for b in selected if '(Tier1)' in b.splitlines()[0]]"
    new = "optimized=[b for b in selected if any(t in b.splitlines()[0] for t in ['(Tier1)', '(Tier1-OSR)'])]"
    assert source.count(old) == 1
    source = source.replace(old, new)
    old = "tier='Tier1',fmas=18"
    new = "tier=block.splitlines()[0].rsplit(' (',1)[1].rstrip(')'),fmas=18"
    assert source.count(old) == 1
    source = source.replace(old, new)
    namespace = dict(__file__=str(Path(__file__).resolve()), __name__='osr_capture_review')
    exec(compile(source, str(original), 'exec'), namespace)
    namespace['capture']()


if __name__ == '__main__': main()
