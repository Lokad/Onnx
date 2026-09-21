"""Reverify the closed reference and failure records, then publish precise scope."""
from common import *
import math

OUT=ROOT/'artifacts/parakeet-stem-reference-final-20260921'
REPORT=Path(__file__).with_name('results-20260921.md')
OBS=Path(__file__).with_name('observations-20260921.json')


def main():
    assert not OUT.exists() and not REPORT.exists() and not OBS.exists()
    spec=read(BASE/'manifest.json');verify(spec);closed=read(BASE/'closed.json');data=read(BASE/'analysis.json')
    assert closed['references_qualified'] and data['references_qualified'] and len(data['agreement'])==33 and len(data['repeats'])==22 and len(data['original'])==8
    assert all(r['failures']==0 for r in data['agreement']) and data['scalar_checks']==9216
    bindings={}
    def check(files):
        for name,expected in files.items():assert pin(ROOT/name)==expected,name;bindings[name]=expected
    check(closed['files']);check(closed['external_files']);bindings[rel(BASE/'closed.json')]=pin(BASE/'closed.json')
    failed_base=ROOT/'artifacts/parakeet-stem-reference-v2-20260921';failed=read(failed_base/'failure-closed.json')
    assert not failed['qualified'];check(failed['files']);check(failed['external_files'])
    bindings[rel(failed_base/'failure-closed.json')]=pin(failed_base/'failure-closed.json')
    unused=ROOT/'artifacts/parakeet-stem-reference-20260921';preparation=read(unused/'preparation-failure.json')
    for name,expected in preparation['files'].items():
        path=unused/name;assert pin(path)==expected;bindings[rel(path)]=expected
    bindings[rel(unused/'preparation-failure.json')]=pin(unused/'preparation-failure.json')
    identities=closed['identities']+failed['identities'];assert len(identities)==9 and all(absent(i) for i in identities)
    # Check original-vs-reference scalar summaries directly from every stem value,
    # including RMS with an independent accurately summed square reduction.
    for row in data['original']:
        original=spec['retained_stems'][row['engine']+'-'+row['input']];a=load_array(ROOT/original['file'],original)
        folder=BASE/'outputs'/(row['reference_engine']+'-'+row['input'])
        record=next(r for r in read(folder/'result.json')['outputs'] if r['name']=='stem');b=load_array(folder/record['file'],record)
        differences=[float(x)-float(y) for x,y in zip(a.reshape(-1),b.reshape(-1),strict=True)]
        errors=[abs(x)/max(1.,abs(float(y))) for x,y in zip(differences,b.reshape(-1),strict=True)]
        assert max(errors)==row['max_scaled'] and sum(x>1e-4 for x in errors)==row['failures']
        assert math.isclose(math.sqrt(math.fsum(x*x for x in differences)/len(differences)),row['rms'],rel_tol=2e-14)
    observations=dict(**data,closure=pin(BASE/'closed.json'),initial_preparation_failure=pin(unused/'preparation-failure.json'),
                      first_worker_failure=pin(failed_base/'failure-closed.json'))
    write(OBS,observations)
    lines=['# Independent Parakeet stem references — 2026-09-21','',
      'Both original float32 engines exceed the unchanged `1e-4` error limit at the',
      'encoder stem on the selected English clip. Independent NumPy and PyTorch',
      'float64 calculations agree across every retained stage within `5.534e-13`',
      'scaled error. Lokad has larger maximum, RMS and failure-count errors on both',
      'feature inputs in this experiment. No production arithmetic, numerical gate',
      'or performance result changes.', '',
      '## Original float32 outputs against the two references','',
      'Each row compares all 75,776 values of the stem output. Inputs are the exact',
      'native and managed feature arrays from the earlier English diagnostic.',
      'Weights and inputs are promoted exactly from their stored float32 values.',
      'The scale is `abs(actual-reference)/max(1,abs(reference))`; RMS is unscaled.', '',
      '| Engine | Feature input | Reference | Maximum scaled error | Values above 1e-4 | RMS error |',
      '|---|---|---|---:|---:|---:|']
    for r in data['original']:
        lines.append(f'| {r["engine"]} | {r["input"]} | {r["reference_engine"]} | {r["max_scaled"]:.10g} | {r["failures"]} | {r["rms"]:.10g} |')
    lines += ['',
      'The managed stem has 200 and 203 failing values with native and managed',
      'features respectively; native ORT has 57 and 69. Both reference engines',
      'give the same counts. Managed RMS is about 3.3 times native RMS on these',
      'two inputs. This does not establish that one engine is always more accurate.', '',
      '## Reference qualification','',
      'The complete 27-node ancestor graph is checked against the declared calculation,',
      'including dependencies, constants, attributes and shape operations. Its numeric',
      'path consists of five convolutions, three ReLUs, a transpose/reshape, a',
      '4,096-by-1,024 matrix projection and bias. NumPy uses explicit windows plus',
      'matrix products or direct depthwise reduction; PyTorch uses grouped `conv2d`',
      'and matrix multiplication. They use separately implemented kernels, with',
      'OpenBLAS and MKL configurations recorded. Neither reference worker loads ORT.', '',
      'Every original float32 input and coefficient is promoted to float64 without',
      'refitting, changing constants or quantizing. All 11 stages are retained for',
      'each of six fresh calls: both feature inputs and one exact repeat per engine.',
      'All 33 full-array reference comparisons pass the predeclared `1e-9` limit;',
      'all 22 repeated arrays match bits. The largest scaled reference difference',
      'is `5.53335155473178e-13`.', '',
      '| Stage | Maximum reference difference across both inputs and repeat | Failing reference values |',
      '|---|---:|---:|']
    for stage in STAGES:
        rows=[r for r in data['agreement'] if r['stage']==stage]
        lines.append(f'| {stage} | {max(r["max_scaled"] for r in rows):.10g} | {sum(r["failures"] for r in rows)} |')
    lines += ['',
      'Each call checks 256 predetermined border/interior coordinates at every',
      'convolution and at the projection, using `math.fsum` of individual products',
      'from that route’s own inputs. The auditor independently reconstructs all',
      '9,216 checks from the saved arrays and original coefficients. Maximum scaled',
      'scalar discrepancy is `9.14823772291129e-14`; all pass `1e-9`. ReLU, reshape',
      'and bias stages also pass exact reconstruction. Five small independent tests',
      'cover grouping, stride-two borders, pointwise convolution and projection layout.', '',
      '## Preserved failures and resources','',
      'An initial test-discovery failure omitted the existing psutil package path;',
      'the corrected invocation passes all five tests. The first preparation then',
      'stops before inference because ONNX’s external-data loader refuses the deliberate',
      'hardlinks used by the preceding trace. Corrected preparation reads only declared,',
      'length-checked slices from the fully hash-verified original sidecar.', '',
      'The first reference worker writes all eleven arrays and 1,536 successful scalar',
      'checks, but its final environment audit rejects a lazily loaded NumPy binary',
      'missing from preparation’s library inventory. It remains failed and is not',
      'reused as qualified reference evidence. The separately frozen correction pins',
      'the complete installed native package inventory and still records/checks each',
      'worker’s actual loaded subset. Arithmetic, schedule, resource limits and gates',
      'remain unchanged; their source equivalence is verified before execution.', '',
      f'The successful phase retains {data["arrays"]} arrays / {data["values"]:,} values /',
      f'{data["bytes"]:,} raw bytes, {data["resource_samples"]} resource samples and seven terminal',
      f'process identities. Its largest sampled RSS is {data["peak_rss"]:,} bytes. The failed',
      'worker’s eleven arrays, ten samples and two terminal identities are separately',
      'retained. All nine recorded identities are verified terminal before publication.',
      'Limits remain 8 GiB RSS, 1 GiB available RAM, 20 GiB free disk and 900 seconds',
      'per worker, with 10 GiB available-memory preflight. Workers run sequentially',
      'on Windows CPU2 with one numerical-library thread. These durations are not',
      'a latency comparison.', '',
      '## Scope and next decision','',
      'These finite independent calculations support a stem accuracy investigation.',
      'They are not a formal correctly-rounded proof or complete encoder/transcription',
      'qualification. The original Windows duration-logit failure remains open.',
      'The [preceding trace](../layer-trace/results-20260921.md) also shows later',
      'encoder error propagation; a stem correction alone is not yet shown sufficient.', '',
      'The next distinct test should capture the original pre-projection input and',
      'compare that exact projection on identical inputs. Its reduction length is',
      '4,096, whereas the five stem convolutions reduce 9 or 256 terms. Longer float32',
      'accumulation is therefore a concrete hypothesis to test, not an attribution',
      'established by the whole-stem references.', '',
      'Source `09524ea`; NumPy 2.2.4, PyTorch 2.11.0+cpu. Original retained managed',
      'outputs use qualified core',
      '`d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4`;',
      'native outputs use ORT 1.29.0. No original float32 inference is rerun here.',
      '[Complete observations](observations-20260921.json) retain every comparison.',
      'Successful closure: `'+pin(BASE/'closed.json')['sha256']+'`.',
      'Final file/report/process verification is retained in',
      '`artifacts/parakeet-stem-reference-final-20260921/verified.json`.', '']
    REPORT.write_text('\n'.join(lines),encoding='utf8')
    bindings[rel(OBS)]=pin(OBS);bindings[rel(REPORT)]=pin(REPORT);bindings[rel(Path(__file__))]=pin(__file__)
    OUT.mkdir();write(OUT/'verified.json',dict(files=bindings,identities=identities,terminal=True,references_qualified=True,
              original_limit=1e-4,reference_limit=REF_LIMIT,scalar_checks=9216,original_comparisons=8,
              failed_worker_retained=True,initial_preparation_failure_retained=True))
    print(json.dumps(dict(files=len(bindings),identities=len(identities),report=rel(REPORT),verification=pin(OUT/'verified.json'))))


if __name__=='__main__':main()
