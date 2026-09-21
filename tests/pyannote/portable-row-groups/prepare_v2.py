"""Guard short reductions after the closed broad probe failed its worst-shape gate."""
import common

failure = common.BASE / 'closed.json'
assert common.pin(failure)['sha256'] == '81e59df4ba5e450c637a04919ec6dbab98f9baf1a6b4e2783b45e43130b4951e'
old = common.read(common.BASE / 'analysis.json')
assert old['passed'] and old['controls_passed'] and not old['eligible']
assert [tuple(r[k] for k in ['m','n','k']) for r in old['rows'] if r['candidate_baseline'] > 1.05] == [(32,9,1440),(32,9,1568)]
common.BASE = common.ROOT / 'artifacts/pyannote-portable-row-groups-v2-20260921'
common.monitor.BASE = common.BASE
path = common.TOOLS / 'prepare.py'
source = path.read_text(encoding='utf8')
assert source.count("shutil.copy2(TOOLS/'Probe.cs',BASE/'consumer/Probe.cs')") == 1
source = source.replace("shutil.copy2(TOOLS/'Probe.cs',BASE/'consumer/Probe.cs')", "shutil.copy2(TOOLS/'ProbeV2.cs',BASE/'consumer/Probe.cs')")
assert source.count("pins[rel(ort)] = pin(ort)") == 1
source = source.replace("pins[rel(ort)] = pin(ort)", "pins[rel(ort)] = pin(ort)\n    predecessor=ROOT/'artifacts/pyannote-portable-row-groups-20260921/closed.json'\n    pins[rel(predecessor)]=pin(predecessor)")
exec(compile(source,str(path),'exec'),dict(__name__='__main__',__file__=str(path)))
