"""Count source-level work for the closed, matched corpus; no model execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
from run import ROOT, SOURCE, APP, MANIFEST, pin, read, write

ATTRIBUTION = ROOT/'artifacts/parakeet-masking-padding-attribution-20260924'
OUTPUT = ROOT/'artifacts/parakeet-masking-padding-work-counts-20260924'


def main():
    assert not OUTPUT.exists()
    closure = read(ATTRIBUTION/'closed.json')
    assert closure['passed'] and closure['analysis'] == pin(ATTRIBUTION/'analysis.json')
    assert pin(ATTRIBUTION/'closed.json')['sha256'] == '0299d2e1d273b2651d45209cb7c9dffb623ae4149cf0cb5eed23eaf89d842e76'
    matched = read(ATTRIBUTION/'analysis.json')
    assert Counter(r['kind'] for r in matched['rows']) == {
        'attention-mask':24,'attention-cleanup':24,'convolution-mask':24,'attention-pad':24,'convolution-pad':24}
    assert all(r['measured_calls'] == 60 and r['shape_observations'] == 19 for r in matched['rows'])
    manifest = read(APP/'collected'/MANIFEST)
    frames = [c['expected']['encoded_frames'] for c in manifest['cases']]
    assert Counter(frames) == {int(k):v for k,v in matched['frames'].items()}
    assert len(frames) == 20 and min(frames) > 0
    receipt = read(SOURCE/'prepared.json'); managed_sources = {}
    for name in [*matched['managed_sources'],'src/Lokad.Onnx/TensorOps.Broadcast.cs','src/Lokad.Onnx/BroadcastedTensor.cs']:
        actual = pin(SOURCE/'source'/name); assert actual == receipt['source'][name]
        if name in matched['managed_sources']: assert actual == matched['managed_sources'][name]
        managed_sources[name] = actual
    native_sources = {}
    for name, wanted in matched['native_sources'].items():
        raw = subprocess.run(['git','-c','gc.auto=0','-C',str(ROOT/'external/onnxruntime'),
            'show',matched['ort_revision']+':'+name],capture_output=True,check=True).stdout
        actual = dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
        assert actual == wanted; native_sources[name] = actual
    where = []
    for family in ['attention-mask','attention-cleanup','convolution-mask']:
        convolution = family == 'convolution-mask'
        conditions = 24*sum(t if convolution else t*t for t in frames)
        outputs = conditions*(1024 if convolution else 8)
        where.append(dict(family=family,condition_values=conditions,output_values=outputs,
            managed_condition_coordinate_steps_at_least=outputs*(3 if convolution else 4),
            ort_x_selection_values=conditions,ort_y_selection_values=outputs,
            ort_logical_float_writes=conditions+2*outputs,
            managed_where_loop_float_writes=outputs))
    pad = []
    for family in ['attention-pad','convolution-pad']:
        convolution = family == 'convolution-pad'
        inputs = 24*sum(1024*t if convolution else 8*t*(2*t-1) for t in frames)
        outputs = 24*sum(1024*(t+8) if convolution else 8*t*2*t for t in frames)
        rows = 24*sum(1024 if convolution else 8*t for t in frames)
        pad.append(dict(family=family,input_values=inputs,output_values=outputs,padding_values=outputs-inputs,
            managed_pad_coordinate_steps=inputs*(3 if convolution else 4),
            source_predicted_ort_innermost_copy_blocks=rows,
            managed_fill_plus_copy_float_writes=outputs+inputs,ort_copy_plus_padding_float_writes=outputs))
    totals = dict(where={k:sum(r[k] for r in where) for k in where[0] if k != 'family'},
        pad={k:sum(r[k] for r in pad) for k in pad[0] if k != 'family'})
    value = dict(passed=True,attribution=pin(ATTRIBUTION/'closed.json'),manifest=pin(APP/'collected'/MANIFEST),
        source_receipt=pin(SOURCE/'prepared.json'),managed_sources=managed_sources,
        ort_revision=matched['ort_revision'],native_sources=native_sources,
        clips=20,passes_counted=1,layers=24,where=where,pad=pad,totals=totals,
        source_level_predictions=True,measured_instruction_counts=False,measured_memory_traffic=False,
        new_inference_calls=0,new_optimization_selected=False,
        caveat='Counts follow source loops and matched logical shapes. They exclude allocation clearing, view materialization, cache effects and compiler transformations. Division and remainder may share a machine instruction. Where condition counts cover at least its required expanded view, not possible additional nested view work. No mask value or actual managed layout is inferred.')
    OUTPUT.mkdir(); write(OUTPUT/'analysis.json',value)
    write(OUTPUT/'closed.json',dict(passed=True,analysis=pin(OUTPUT/'analysis.json'),analyzer=pin(__file__),new_inference_calls=0))
    print(json.dumps(dict(closure=pin(OUTPUT/'closed.json'),where=where,pad=pad,totals=totals)))


if __name__ == '__main__': main()
