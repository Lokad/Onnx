"""Keep every existing compiler warning and reject added or suppressed warnings."""
from collections import Counter
from pathlib import Path
import re
from protocol import pin

ROOT=Path(__file__).resolve().parents[3]
REFERENCE=ROOT/'artifacts/parakeet-observed-dense-where-root-amd-v2-20260924/collected'


def census(folder):
    rows=Counter()
    for path in (folder/'logs').iterdir():
        if path.suffix not in ['.stdout','.stderr'] or path.name.startswith(('bridge-restore.','bridge-build.')):continue
        for line in path.read_text(encoding='utf8').splitlines():
            if re.search(r'\bwarning\b',line,re.IGNORECASE):
                normalized=re.sub(r'/dev/shm/lokad-[^/\s]+/','CAMPAIGN/',line.strip())
                rows[(path.name,normalized)]+=1
    return rows


def compare(folder,reference=REFERENCE):
    for path in (folder/'logs').iterdir():
        if path.name.startswith(('bridge-restore.','bridge-build.')):
            assert path.suffix in ['.stdout','.stderr'] or path.suffix=='.jsonl'
            if path.suffix in ['.stdout','.stderr']:
                warnings=[line.strip() for line in path.read_text(encoding='utf8').splitlines() if re.search(r'\bwarning\b',line,re.IGNORECASE)]
                assert all(line=='0 Warning(s)' for line in warnings),warnings
    before,after=census(reference),census(folder)
    assert before==after,dict(missing=list((before-after).items()),added=list((after-before).items()))
    rows=[dict(log=name,text=text,count=count) for (name,text),count in sorted(after.items())]
    diagnostics=sum(r['count'] for r in rows if ': warning CS8604:' in r['text'])
    assert diagnostics==4
    return dict(passed=True,no_new_warning=True,existing_cs8604_occurrences=diagnostics,
        source_warnings=2,summary_repeats_included=True,rows=rows,
        reference_log=pin(reference/'logs/cli-build.stdout'))
