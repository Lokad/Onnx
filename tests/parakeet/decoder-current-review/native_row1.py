"""Identify the executed one-row native GEMM branch from retained samples only."""
from bisect import bisect_right
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from analyze import ROOT,OUT,pin,read

BASE=ROOT/'artifacts/parakeet-decoder-native-row1-review-20260927'
SAMPLES=ROOT/'artifacts/parakeet-ort-native-samples-20260924'
MATCH=ROOT/'artifacts/parakeet-ort-native-kernel-match-20260924'
ACTIVATION=ROOT/'artifacts/parakeet-ort-activation-review-20260926'
START,END=0x12491b0,0x124939a


def analyze():
    inputs={}
    for folder,digest in [(SAMPLES,'546b3a58af8814772f0a5b816dbcbece710facc01980acebeb2fc93d7f0dcca9'),
        (MATCH,'54c791e4c46e511c52b615909dc1249dae0f96d7c4734c174a5e12aeabc6d414'),
        (ACTIVATION,'41f851c4e473856383c6d51f7b5a0e86d5f0ff8c47947df1f4176b35e2301fc3')]:
        proof=read(folder/'closed.json');assert proof['passed'] and pin(folder/'closed.json')['sha256']==digest
        wanted=proof['files']['analysis.json'] if folder==ACTIVATION else proof['analysis']
        assert wanted==pin(folder/'analysis.json')
        inputs[folder.name]=dict(closure=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'))
    original=read(SAMPLES/'analysis.json');terminal=read(SAMPLES/'terminal.json')
    spec=read(MATCH/'prepared.json');match=read(MATCH/'analysis.json')
    assert spec['revision']=='2e2543fbe9fae542f921d47a72d21d5a4ef0b710'
    for name,wanted in spec['files'].items():assert pin(MATCH/name)==wanted
    binary_path=ACTIVATION/'onnxruntime_pybind11_state.so'
    assert pin(binary_path)==spec['binary']==original['loaded_binary']
    binary=binary_path.read_bytes()
    assert (MATCH/'candidate.bin').read_bytes()==(MATCH/'original.bin').read_bytes()==binary[match['start']:match['end']]
    assert match['bytes']==8904 and match['start']<=START<END<match['end']
    # The exact matched function branches to this body only for CountM == 1.
    assert binary[0x1248483:0x1248487]==bytes.fromhex('49 83 f8 01')
    assert binary[0x1248487:0x124848d]==bytes.fromhex('0f 84 23 0d 00 00')
    asm=(MATCH/'FgemmKernelAvx512FCommon.h').read_text()
    assert '.LProcessCountM1:\n        ProcessCountM 1' in asm
    loader=importlib.util.spec_from_file_location('row1_disassembly',ROOT/'tests/parakeet/ort-activation-review/analyze.py')
    helper=importlib.util.module_from_spec(loader);loader.loader.exec_module(helper)
    listing,instructions=helper.disassemble(binary_path,START,END,binary)
    loop={k:v for k,v in instructions.items() if 0x12491c7<=k<0x1249270}
    assert sum('vfmadd231ps' in r['instruction'] for r in loop.values())==8
    assert sum('vbroadcastss' in r['instruction'] for r in loop.values())==4
    assert not any('ymm' in r['instruction'] for r in loop.values())
    assert not any(r['instruction'].startswith('call') for r in instructions.values())
    sys.path.insert(0,str(ROOT/'tests/parakeet/ort-diagnosis-amd'))
    loader=importlib.util.spec_from_file_location('original_row1_parser',SAMPLES/'audit-final.py')
    parser=importlib.util.module_from_spec(loader);loader.loader.exec_module(parser)
    for name in ['perf.data','perf.script','target.maps','requests/result.json']:
        path=SAMPLES/'collected'/name;assert pin(path)==terminal['files'][name]
        inputs[path.relative_to(ROOT).as_posix()]=pin(path)
    raw,addresses=parser.raw_records(SAMPLES/'collected/perf.data')
    assert json.loads(json.dumps(raw))==original['raw'] and raw['lost']==0
    inspected=read(SAMPLES/'native-inspection.json');maps=[]
    for line in (SAMPLES/'collected/target.maps').read_text().splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)==6 and fields[5]==inspected['path'] and 'x' in fields[1]:
            first,last=[int(s,16) for s in fields[0].split('-')];maps.append((first,last,int(fields[2],16)))
    (first,last,file_offset),=maps
    windows=sorted((r['start_ticks'],r['end_ticks']) for r in read(SAMPLES/'collected/requests/result.json')['records'] if r['phase']=='measured')
    starts=[a for a,b in windows];assert len(windows)==60
    selected=[];seen=set();measured=weight=0
    for sample in parser.samples(SAMPLES/'collected/perf.script'):
        key=sample['pid'],sample['tid'],sample['stamp'];assert key in addresses and key not in seen;seen.add(key)
        index=bisect_right(starts,sample['stamp'])-1
        if index<0 or sample['stamp']>=windows[index][1]:continue
        measured+=1;weight+=sample['period'];ip=addresses[key]
        if not first<=ip<last:continue
        offset=ip-first+file_offset
        if not START<=offset<END:continue
        assert offset in instructions and sample['frames'][0]['ip']==ip
        callers=[f for f in sample['frames'][1:] if f['dso']==inspected['path']]
        selected.append(dict(request=index,offset=hex(offset),period_ns=sample['period'],
                             stamp=sample['stamp'],tid=sample['tid'],frames=len(sample['frames']),native_callers=len(callers)))
    assert seen==set(addresses) and measured==original['measured_samples'] and weight==original['measured_period_ns']
    assert selected and all(r['native_callers']==0 for r in selected)
    period=sum(r['period_ns'] for r in selected)
    value=dict(passed=True,new_inference_calls=0,new_compilation=False,inputs=inputs,
        binary=pin(binary_path),source_revision=spec['revision'],routine_bytes=match['bytes'],
        branch=dict(start=hex(START),end=hex(END),comparison='0x1248483: cmp r8,1',jump='0x1248487: je 0x12491b0',
                    vector_bits=512,main_output_columns=32,main_accumulators=2,reduction_unroll=4,fmas_per_main_iteration=8),
        samples=selected,sample_count=len(selected),request_count=len({r['request'] for r in selected}),
        instruction_counts=dict(Counter(r['offset'] for r in selected)),period_ns=period,
        estimated_seconds_per_corpus=period/3e9,measured_samples=measured,measured_period_ns=weight,
        every_sample_reconciled=True,per_node_attribution=False,argument_capture=False,
        complete_projection_cost=False,product_changed=False)
    return value,listing


def main():
    assert sys.argv[1:] in [[],['--publish']]
    value,listing=analyze();target=OUT/'native-row1-20260927.json'
    if not sys.argv[1:]:assert read(target)==value
    else:
        assert not BASE.exists() and not target.exists();BASE.mkdir()
        raw=json.dumps(value,indent=2,allow_nan=False)+'\n'
        (BASE/'analysis.json').write_text(raw,encoding='utf8');target.write_text(raw,encoding='utf8')
        (BASE/'disassembly.txt').write_text(listing,encoding='utf8')
        proof=dict(passed=True,analysis=pin(BASE/'analysis.json'),disassembly=pin(BASE/'disassembly.txt'),
                   analyst=pin(Path(__file__)),inputs=value['inputs'],new_inference_calls=0)
        (BASE/'closed.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in value.items() if k not in ['inputs','samples','instruction_counts']}))


if __name__ == '__main__':main()
