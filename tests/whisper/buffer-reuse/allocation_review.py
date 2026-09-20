"""Account for fixed shapes and logits copying in the sealed 20-request worker; no inference."""
from pathlib import Path
import hashlib,json

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/whisper-allocation-review-20260920'


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def main():
    snapshot=json.loads((BASE/'conformance-snapshot.json').read_text())
    assert snapshot['run']['complete'] and snapshot['run']['code']==0
    assert pin(BASE/'conformance-result.json')==snapshot['gate']['worker']==snapshot['result_pin']
    value=json.loads((BASE/'conformance-result.json').read_text());assert value==snapshot['result']
    prototype=ROOT/'artifacts/whisper-buffer-reuse-20260920';frozen=json.loads((prototype/'frozen.json').read_text())
    assert pin(prototype/'frozen.json')==snapshot['gate']['frozen']
    assert value['core_sha256']==frozen['files']['bin/Lokad.Onnx.dll']['sha256']
    assert value['data_sha256']==frozen['files']['bin/Lokad.Onnx.Data.dll']['sha256']
    manifest=json.loads((prototype/'prototype-manifest.json').read_text());assert pin(prototype/'prototype-manifest.json')['sha256']==value['manifest_sha256']
    config=ROOT/'models/whisper-large-v3-turbo/config.json'
    assert pin(config)=={k:manifest['models']['config.json'][k] for k in ['bytes','sha256']}
    cfg=json.loads(config.read_text());assert cfg['d_model']==1280 and cfg['decoder_layers']==4 and cfg['decoder_attention_heads']==20
    generation=prototype/'source/src/Lokad.Onnx.Data/WhisperGeneration.cs'
    source=json.loads((prototype/'source.json').read_text());assert pin(generation)==source['files']['src/Lokad.Onnx.Data/WhisperGeneration.cs']
    text=generation.read_text();assert 'VocabularySize = 51866' in text and 'prefixLength = timestamps ? 3 : 4' in text
    assert 'var logits = RequireFloat(outputs, "logits", new[] { 1, ids.Length, VocabularySize }).ToArray();' in text
    cross=4*2*20*1500*64*4
    first_self=4*2*20*4*64*4
    first_logits=4*51866*4
    assert (cross,first_self,first_logits)==(61440000,163840,829856)
    rows=[]
    for index,(row,case) in enumerate(zip(value['records'],manifest['cases'],strict=True)):
        assert row['name']==case['name'] and row['result']==case['expected'] and row['ownership'] is True
        n=len(row['result']['token_ids']);assert 1<=n<=444
        # DecodeCore creates one independent ToArray per model call: four rows
        # on the first call, one row on each subsequent call, including EOS.
        logits_copy=(4+n-1)*51866*4
        # These are logical exported output sizes, not an allocation attribution:
        # views, aliases and additional graph copies must be measured separately.
        self_payload=sum(4*2*20*(4+step)*64*4 for step in range(n))
        if index:
            assert row['pools']['encodingExecution']['allocated_new_bytes']==7680000
            assert row['pools']['firstExecution']['allocated_new_bytes']==cross+first_self+first_logits
        rows.append(dict(name=row['name'],tokens=n,public_allocated_bytes=row['allocated_bytes'],
            logits_copy_payload_bytes=logits_copy,logical_decoder_self_output_bytes=self_payload,
            first_decoder_new_pool_bytes=row['pools']['firstExecution']['allocated_new_bytes']))
    warm=rows[1:];allocation=sum(r['public_allocated_bytes'] for r in warm)
    copies=sum(r['logits_copy_payload_bytes'] for r in warm)
    # A single four-row scratch array per request would still allocate 829856 B.
    possible=sum(r['logits_copy_payload_bytes']-first_logits for r in warm)
    result=dict(passed=True,scope='Completed conformance only; endurance remains separate',rows=rows,
        counts=dict(calls=20,warm_calls=19),first_decoder_output_payload=dict(cross_attention=cross,self_attention=first_self,logits=first_logits,total=cross+first_self+first_logits),
        warm_public_allocated_bytes=allocation,warm_logits_copy_payload_bytes=copies,logits_copy_fraction=copies/allocation,
        hypothetical_per_request_scratch_payload_reduction=possible,hypothetical_fraction=possible/allocation,
        caveat='Payload accounting excludes array headers and all other allocations. The scratch estimate is not an implemented or measured saving. Logical self-cache output sizes are not fresh allocation attribution.',
        inputs={p.relative_to(ROOT).as_posix():pin(p) for p in [BASE/'conformance-result.json',BASE/'conformance-snapshot.json',prototype/'frozen.json',prototype/'prototype-manifest.json',config,generation]})
    with (BASE/'review.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','inputs']}))


if __name__=='__main__':main()
