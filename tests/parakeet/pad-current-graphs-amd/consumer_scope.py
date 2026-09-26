"""Reuse the graph protocol, adding only the already qualified short-e5 route."""
from pathlib import Path
import hashlib

TOOLS=Path(__file__).resolve().parent
ROOT=TOOLS.parents[2]
PARENT=TOOLS.parent/'packed-final-row-graphs-amd'
SHORT=ROOT/'tests/benchmarks/e5-steady-short-amd'
UNCHANGED=['protocol.py','checks_e5.py','native.py','native-e5.py','statistics_base.py',
           'statistics_e5.py','test_statistics.py','test_e5_statistics.py','test_inventory.py']
COPIES={'short_checks.py':'checks.py','native-short.py':'native.py',
        'statistics_short.py':'statistics.py'}


def replacements(name):
    if name=='remote.py':return [
        ("native='native-e5.py' if case=='e5-30tok' else 'native.py'",
         "native='native-short.py' if case=='e5-8tok' else 'native-e5.py' if case=='e5-30tok' else 'native.py'"),
        ("runtimes='runtimes-e5' if case=='e5-30tok' else 'runtimes'",
         "runtimes='runtimes-short' if case=='e5-8tok' else 'runtimes-e5' if case=='e5-30tok' else 'runtimes'")]
    if name=='checks.py':return [
        ("consumer_key='e5_consumer' if key=='e5-30tok' else 'consumer'",
         "consumer_key='short_consumer' if key=='e5-8tok' else 'e5_consumer' if key=='e5-30tok' else 'consumer'"),
        ("native_script='native-e5.py' if key=='e5-30tok' else 'native.py'",
         "native_script='native-short.py' if key=='e5-8tok' else 'native-e5.py' if key=='e5-30tok' else 'native.py'"),
        ("warmup=1200 if key=='e5-30tok' else 600", "warmup=6000 if key=='e5-8tok' else 1200 if key=='e5-30tok' else 600")]
    if name=='statistics.py':return [
        ('from protocol import CASES', 'from statistics_short import summarize as short_e5\nfrom protocol import CASES'),
        ("(warmed_e5 if keys=={'e5-30tok'} else baseline)", "(short_e5 if keys=={'e5-8tok'} else warmed_e5 if keys=={'e5-30tok'} else baseline)")]
    if name=='run.py':return [
        ('parakeet-packed-final-row-graphs-amd-20260925','parakeet-pad-current-graphs-amd-20260926'),
        ('parakeet-packed-final-row-graphs-20260925','parakeet-pad-current-graphs-20260926'),
        ('def observe():\n', "def observe():\n    assert not (BASE/'closed.json').exists(), 'Preserve the closed campaign'\n"),
        ("'runtimes','runtimes-e5','source'", "'runtimes','runtimes-e5','runtimes-short','source'")]
    if name=='audit.py':return [
        ('from statistics import summarize', 'from statistics import summarize\nfrom reuse import review as review_consumers'),
        ("name.startswith(('runtimes/','runtimes-e5/'))", "name.startswith(('runtimes/','runtimes-e5/','runtimes-short/'))"),
        ('    reports={};resources=[];clocks=[]', '    short_consumer=review_consumers(c,spec,built)[\'short\']\n    reports={};resources=[];clocks=[];setups=[]'),
        ("        clocks.extend(dict(process=name,**clock) for clock in v['clocks'])",
         "        clocks.extend(dict(process=name,**clock) for clock in v['clocks'])\n        setups.append(dict(process=name,seconds=v['setup_seconds']))"),
        ('assert len(clocks)==41112', 'assert len(setups)==72 and len(clocks)==73512'),
        ('e5_consumer=e5_consumer,clocks=len(clocks),measured=8640,root_product_changed=False',
         "e5_consumer=e5_consumer,short_consumer=short_consumer,clocks=len(clocks),measured=8640,setups=setups,products=spec['products'],root_product_changed=False")]
    raise AssertionError(name)


def expected(name):
    text=(PARENT/name).read_text(encoding='utf8')
    for before,after in replacements(name):
        assert text.count(before)==1,(name,before)
        text=text.replace(before,after)
    return text


def verify_scope():
    sources=[]
    for name in UNCHANGED:
        assert (TOOLS/name).read_bytes()==(PARENT/name).read_bytes(),name
        sources.append(PARENT/name)
    for target,original in COPIES.items():
        assert (TOOLS/target).read_bytes()==(SHORT/original).read_bytes(),target
        sources.append(SHORT/original)
    for name in ['remote.py','checks.py','statistics.py','run.py','audit.py']:
        assert (TOOLS/name).read_text(encoding='utf8')==expected(name),name
        sources.append(PARENT/name)
    return {p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources}


if __name__=='__main__':print(verify_scope())
