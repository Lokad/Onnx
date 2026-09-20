"""Audit and retain the private helper's contracts and actual prepared-storage inspection."""
from pathlib import Path
import json,shutil,sys,xml.etree.ElementTree as ET
from prepare import ROOT,BASE,pin,write
sys.path.insert(0,str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil


def main():
    assert not (BASE/'local-closed.json').exists()
    built=json.loads((BASE/'built.json').read_text());assert built['tests_passed']
    source=json.loads((BASE/'source.json').read_text())
    for name,wanted in source['files'].items():assert pin(BASE/'source'/name)==wanted,name
    for name,wanted in source['inputs'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in built['files'].items():assert pin(BASE/name)==wanted,name
    folder=BASE/'inspection-v2';state=json.loads((folder/'identity.json').read_text());value=json.loads((folder/'result.json').read_text())
    assert state['complete'] and state['code']==0 and 'error' not in state and value['passed'] and not value['inference']
    try:assert psutil.Process(state['child']['pid']).create_time()!=state['child']['birth']
    except psutil.NoSuchProcess:pass
    for name,wanted in state['files'].items():assert pin(folder/name)==wanted,name
    assert pin(folder/'result.json')==state['result']
    samples=[json.loads(s) for s in (folder/'samples.jsonl').read_text().splitlines()]
    assert len(samples)==state['samples'] and state['preflight_available']>=state['limits']['preflight']
    assert 0<state['seconds']<state['limits']['seconds']
    assert state['peak_rss']==max(s['rss'] for s in samples)
    previous=0
    for row in samples:
        assert previous<=row['seconds']<state['seconds'];previous=row['seconds']
        assert row['rss']<state['limits']['rss'] and row['available']>=state['limits']['available'] and row['affinity']==[2]
    assert value['affinity']==4 and value['processor_count']==1 and value['flags']=={} and value['runtime']=='.NET 10.0.12'
    for key,name in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll')]:
        assert value[key]==pin(BASE/'product-bin'/name)['sha256']==pin(BASE/'source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0'/name)['sha256']
    census=json.loads((BASE/'weight-census.json').read_text())
    expected=sum(row['bytes'] for row in census['graphs'][1]['rows'] if row.get('exact_match_in_first') and row['type']==1 and row['bytes']>=4096)
    assert expected==635187200==value['logical_shared_bytes']
    total=0
    for name in ['first','past']:
        assert value['before'][name]==value['after'][name]
        total+=sum(r['bytes'] for r in value['before'][name]['initializers'])
    assert total==value['before']['unique_payload_bytes']==1985006252
    assert total-expected==value['after']['unique_payload_bytes']==1349819052
    assert value['before']['unique_arrays']-value['after']['unique_arrays']==88
    trx=ET.parse(BASE/'test-results/backend.trx');ns={'t':'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
    counters=trx.find('.//t:Counters',ns).attrib;assert counters['passed']=='3101' and counters['failed']=='0' and counters['total']=='3194'
    new=[r for r in trx.findall('.//t:UnitTestResult',ns) if r.attrib['testName'].startswith('Lokad.Onnx.Backend.Tests.WhisperDecoderWeightsTests.')]
    assert len(new)==12 and all(r.attrib['outcome']=='Passed' for r in new)
    summary=dict(passed=True,private_prototype=True,inference=False,tests=dict(passed=3101,skipped=93,new=12),
        logical_shared_bytes=expected,unique_initializer_bytes_before=total,unique_initializer_bytes_after=total-expected,
        arrays_before=value['before']['unique_arrays'],arrays_after=value['after']['unique_arrays'],
        resource=dict(samples=len(samples),peak_rss=state['peak_rss'],min_available=min(s['available'] for s in samples),seconds=state['seconds']),
        result=state['result'],child=state['child'],scope='Actual prepared initializer storage and local contracts only; no inference, RSS saving or production qualification claimed')
    tracked=Path(__file__).resolve().parent;write(tracked/'local-observations-20260920.json',summary)
    report=f'''# Private Whisper decoder-weight sharing: local proof

The prototype shares **635,187,200 bytes** of identical initializer storage
between the first and past decoders inside one transcriber. An actual model
preparation check reduces unique initializer backing storage from
**1,985,006,252 to 1,349,819,052 bytes**, with **587 to 499 arrays**. These are
referenced initializer payloads, not total managed heap or process RSS savings.

All initializer names, tensor names, types, full shapes and full-content hashes
remain identical before and after sharing, including prepared folded and packed
entries. Node names/operators/input/output bindings and retained packed-byte
totals also match. The independent serialized-model census predicts exactly the
observed shared payload after applying the helper's FP32/4 KiB eligibility rule.
It preserves independent tensor wrappers and each graph's own preparation state.

The complete backend suite passes **3,101 tests**, with **93 skipped**. Its twelve
new cases check backing-array identity, separate metadata, signed-zero and NaN
bits, shape/layout/size exclusions, observable inputs/outputs, prepared MatMul
rebuilding, exact outputs, held-output lifetime and input preservation. General
graph contracts and production source are unchanged; the helper is used only by
the separate private Whisper prototype before execution contexts are created.

The inspection runs on Windows/.NET 10.0.12, CPU 2, with no runtime overrides or
native ORT. It performs model import/preparation and byte hashing, **no inference**.
All {len(samples)} resource samples pass a 120-second, 8 GiB RSS and 4 GiB
available-memory guard. Peak observed RSS is {state['peak_rss']:,} bytes; the
original process identity is terminal. That one-process peak is not a before/after
RSS comparison. No explicit garbage collection is used by the .NET consumer.

The helper/product build has zero warnings/errors. The test build retains four
nullable warnings in unresolved-binding fixtures. The inspection build retains
two platform-analysis warnings for its Windows affinity check. Its first build
failed on generic inference for three `MemoryMarshal.TryGetArray` calls; explicit
type parameters fix the consumer in a new `inspection-v2` directory. An earlier
launcher import found psutil absent from global Python and now uses the already
installed isolated psutil 7.0.0. No global installation or model rerun occurred.

The private source inherits the independently tested buffer-reuse prototype and
adds this distinct storage change. Its AMD application, numerical and repeated
request resource qualification remains pending. The existing buffer-reuse
endurance process is a different immutable artifact and is not modified by this
work. Existing memory/numerical failures remain unchanged.

[Summary observations](local-observations-20260920.json) record exact counters,
binary result identity and resource bounds. Full source pins, source diff, both
inspection build attempts, successful tests, every model-initializer hash and
resource samples are under `artifacts/whisper-weight-sharing-20260920`.
'''
    with (tracked/'local-results-20260920.md').open('x',encoding='utf-8') as f:f.write(report)
    snapshots=BASE/'local-tool-snapshots';snapshots.mkdir()
    for p in sorted(tracked.iterdir()):
        if p.is_file():shutil.copyfile(p,snapshots/p.name)
    files={p.relative_to(ROOT).as_posix():pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()}
    for name in ['local-results-20260920.md','local-observations-20260920.json']:files[(tracked/name).relative_to(ROOT).as_posix()]=pin(tracked/name)
    write(BASE/'local-closed.json',dict(passed=True,inference=False,private_prototype=True,files=files,child=state['child']))
    for name,wanted in files.items():assert pin(ROOT/name)==wanted,name
    print(json.dumps(dict(passed=True,closed=pin(BASE/'local-closed.json'),pins=len(files),summary=summary)))


if __name__=='__main__':main()
