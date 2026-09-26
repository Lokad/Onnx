"""Route each complete case to its unchanged, previously qualified scorer."""
from statistics_base import summarize as baseline
from statistics_e5 import summarize as warmed_e5
from statistics_short import summarize as short_e5
from protocol import CASES


def summarize(reports):
    keys={v['key'] for v in reports.values()}
    assert len(keys)==1 and keys.issubset(CASES)
    return (short_e5 if keys=={'e5-8tok'} else warmed_e5 if keys=={'e5-30tok'} else baseline)(reports)
