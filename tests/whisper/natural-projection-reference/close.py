"""Render complete numerical observations and bind immutable terminal evidence."""
import argparse,datetime
from common import *

def render(value,spec):
    worst=max(r['local_error']['max_scaled'] for r in value['rows']);failed=sum(r['local_error']['failed_values'] for r in value['rows'])
    text=['# Natural Whisper projection references — September 20, 2026','',
        f"All {value['reference_arrays']} float64 references were computed from the actual saved normalized inputs and original float32 weights. The largest FP32 projection error against its own-input reference is **{worst:.9g}** under `abs(actual-reference)/max(1,abs(reference))`; **{failed} values** exceed the existing 1e-4 limit. Neither FP32 engine is treated as mathematical truth.",'',
        'These are numerical diagnostics of saved Windows layer-20 arrays, not new model inference or latency measurements. They cover all three selected natural clips, both feature sources, all four engine/incoming-producer combinations and the first-clip repeat. Every frame, including padding, is included. The original full-encoder numerical failures remain open.','',
        '| Request | Features | Projection | Actual MM−NN | Propagated input difference | Local rounding difference | Local / actual L2 |',
        '|---|---|---|---:|---:|---:|---:|']
    for group in value['groups']:
        c=group['contrasts']['MM-NN'];ratio=c['local_residual_difference']['l2']/c['actual']['l2'] if c['actual']['l2'] else 0
        label=group['name']+(' (repeat)' if group['request']>=6 else '')
        text.append(f"| {label} | {group['features']} | {group['projection']} | {c['actual']['max_scaled']:.9g} | {c['propagated']['max_scaled']:.9g} | {c['local_residual_difference']['max_scaled']:.9g} | {ratio:.6f} |")
    text+=['',
        'Here MM is the managed cut on managed incoming state and NN is the native cut on native incoming state; the feature source is separate. Each projection uses its own saved normalized intermediate. The signed decomposition is `actual_MM−actual_NN = reference_MM−reference_NN + (actual_MM−reference_MM)−(actual_NN−reference_NN)`. The three maximum-error columns share `max(1,abs(actual_NN))`; their separate maxima need not occur at the same coordinate or add up. The L2 ratio compares norms, not a causal percentage.','',
        '[Complete observations](observations-20260920.json) include all 64 own-input comparisons, five contrasts for each request/projection (diagonal, both same-incoming and both within-engine input effects), absolute/scaled maxima and coordinates, RMS/L2, every failing-value count and decomposition closure residual. All repeat reference arrays match their first-request bytes.','',
        f"Every reference uses float64 matrix multiplication, with exact promotion of the original finite float32 inputs/weights and the original fc1 bias. A conservative IEEE-arithmetic accumulation bound uses gamma_(2n+2) and an upward-adjusted absolute-product sum. The largest observed bound is {max(r['reference_error_bound_max'] for r in value['rows']):.9g} absolute. Independent Python math.fsum checks cover {value['scalar_checks']} coordinates, including a fixed spread and each local-error maximum; the largest difference is {value['max_fsum_error']:.9g}. This checks selected coordinates independently, not every dot product with arbitrary precision. The complete arrays and their metrics are retained.",'',
        f"One local CPU2 worker uses NumPy {spec['numpy']} / OpenBLAS 0.3.28 with one BLAS thread, under a CPU0 supervisor. It completed in {value['resources']['seconds']:.3f} seconds, with {value['resources']['samples']} resource samples, {value['resources']['peak_sampled_rss']:,} bytes peak sampled RSS and {value['resources']['minimum_available']:,} bytes minimum available memory. Both original process identities are terminal. Sampling does not establish an absolute peak.",'',
        'The saved inference source is the [closed layer-20 crossed-input diagnostic](../layer20-cross/results-20260920.md), original core c6bf781/.NET10.0.12 and Microsoft ORT1.29.0. Current e5 product changes are unrelated to these saved arrays. The diagnostic isolates two affine stages; it does not locate or resolve all earlier encoder differences.','',
        f"New diagnostic source `{spec['source_revision']}`; original corrected receipt SHA256 `{spec['prior_receipt']}`. Manifest SHA256 `{value['manifest']['sha256']}`. All {value['reference_bytes']:,} reference bytes, full scalar checks, source/model identities and resource records are retained in `artifacts/whisper-natural-projection-reference-20260920`.",'']
    return '\n'.join(text)

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);args=p.parse_args();base=Path(args.artifact).resolve()
    value=read(base/'audit.json');spec=read(base/'manifest.json');state=read(base/'campaign.json')
    assert value['passed'] and value['manifest']==pin(base/'manifest.json') and absent(state['worker']) and absent(state['supervisor']);verify(spec['files'])
    folder=Path(__file__).parent;report=folder/'results-20260920.md';observations=folder/'observations-20260920.json'
    with report.open('x',encoding='utf-8') as stream:stream.write(render(value,spec))
    write(observations,value)
    receipt=dict(passed=True,scope='Numerical reference diagnostic; original full-model numerical gate remains open',closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        births=[state['supervisor'],state['worker']],reports={rel(p):pin(p) for p in [report,observations]},
        files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()})
    write(base/'closed.json',receipt);print(json.dumps(dict(receipt=pin(base/'closed.json'),files=len(receipt['files']),reports=receipt['reports'])))

if __name__=='__main__':main()
