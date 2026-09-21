"""Condition the complete workload before timing; preserve the guarded failed trial."""
import common

prior=common.ROOT/'artifacts/pyannote-portable-row-groups-v2-20260921'
assert common.pin(prior/'closed.json')['sha256']=='a057343ed1d63166f178eb29f241bcb21d087d14f5f0614ea60984e112b94a97'
analysis=common.read(prior/'analysis.json')
assert analysis['passed'] and analysis['controls_passed'] and not analysis['eligible']
assert [tuple(r[k] for k in ['m','n','k']) for r in analysis['rows'] if r['candidate_baseline']>1.05]==[(32,288,160)]
common.BASE=common.ROOT/'artifacts/pyannote-portable-row-groups-v3-20260921'
common.monitor.BASE=common.BASE
path=common.TOOLS/'prepare.py';source=path.read_text(encoding='utf8')
assert source.count("shutil.copy2(TOOLS/'Probe.cs',BASE/'consumer/Probe.cs')")==1
source=source.replace("shutil.copy2(TOOLS/'Probe.cs',BASE/'consumer/Probe.cs')","shutil.copy2(TOOLS/'ProbeV3.cs',BASE/'consumer/Probe.cs')")
assert source.count("pins[rel(ort)] = pin(ort)")==1
source=source.replace("pins[rel(ort)] = pin(ort)","pins[rel(ort)] = pin(ort)\n    predecessor=ROOT/'artifacts/pyannote-portable-row-groups-v2-20260921/closed.json'\n    pins[rel(predecessor)]=pin(predecessor)")
exec(compile(source,str(path),'exec'),dict(__name__='__main__',__file__=str(path)))
