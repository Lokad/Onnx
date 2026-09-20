"""Close fully audited hardware arithmetic/code qualification, without timing claims."""
from pathlib import Path
import argparse,datetime,json
from common import pin,read,write,verify
from vm import ssh,prefix

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args();base=args.artifact.resolve();payload=base/'collected'
    assert not (base/'closed.json').exists();verify(payload);audit=read(base/'audit.json');collection=read(base/'collection-check.json')
    assert collection['passed'] is True and collection['streamed'] is True and collection['archive']==pin(base/'results.tar.gz')
    expected=collection['remote']['collection']['files']|{'collection.json':collection['remote']['receipt']}
    assert {p.relative_to(payload).as_posix() for p in payload.rglob('*') if p.is_file()}==set(expected)
    for name,want in expected.items():assert pin(payload/name)==want,name
    assert audit['passed'] is True and audit['bundle']==pin(payload/'bundle.json') and audit['code']['passed'] is True
    assert {(r['pid'],r['birth']) for r in audit['births']}=={(r['pid'],r['birth']) for r in collection['remote']['collection']['births']}
    review=read(base/'code-review.json');assert review['passed'] is True and review['jit']==pin(payload/'result/code/jit.txt') and review['code']==audit['code']
    terminal=json.loads(ssh(prefix()+'births=%r\n'%audit['births']+'''
for item in births:
 try:assert psutil.Process(item['pid']).create_time()!=item['birth'],item
 except psutil.NoSuchProcess:pass
print(json.dumps(dict(checked_at=time.time(),births=births)))
'''))
    report=Path(__file__).with_name('results-20260920.md');assert not report.exists()
    lines=['# Wider LayerNorm: AMD arithmetic and generated code — 2026-09-20','',
      'The wider final-transform prototype passes actual AMD arithmetic and generated-code qualification. '
      'Both the clean-runtime and declared-disassembly phases complete all **915 cases**, **32,182,096 exact comparisons** and **125 real LayerNorm captures**. '
      'This establishes eligibility for a separate complete-bank timing experiment; it does not establish an e5 or ORT speedup and changes no production default.','',
      'Each phase independently verifies 15,475,200 saved real output values against scalar centered-double normalization, '
      'reconstructs all 790 synthetic input cases and preserves output guards, caller inputs, parameters and in-place behavior. '
      'The original full e5 captures and their model/native checks remain bound to the [closed local proof](../layernorm-output/results-20260920.md).','',
      'The candidate widens only the final double transform to sixteen floats through two 512-bit double vectors. '
      'Centered mean/variance, arithmetic association and original vector/scalar tails remain unchanged. '
      'Both candidate and original copy match the actual archived product kernel bit for bit throughout the exercised cases.','',
      'Linux .NET 10.0.8 on the exclusive AMD VM, logical CPU 2 inherited before CLR startup; supervisor CPU 0. '
      'The ordinary product vector width is eight floats, with hardware Vector512 and AVX-512 enabled. '
      'The clean phase has no runtime overrides. The code phase declares only its method-disassembly filter and output file. '
      'Its subsequent three-second full-kernel warmup observes tiered code generation and supplies no latency comparison.','',
      f"Observed product body: `{audit['code']['product_header']}` ({audit['code']['product_code_bytes']} bytes).",
      f"Observed candidate body: `{audit['code']['wide_header']}` ({audit['code']['wide_code_bytes']} bytes).",'',
      'The inspected optimized candidate uses zmm double subtraction, multiplication and addition plus float/double conversions. '
      'The original uses ymm double arithmetic. No fused multiply-add substitution is accepted; all emitted versions remain available.','',
      '| Phase | Resource samples | Peak group RSS bytes | Minimum available bytes | Maximum independent scaled error |',
      '|---|---:|---:|---:|---:|']
    for phase,value in audit['phases'].items():
        resource=value['resources'];lines.append(f"| {phase} | {resource['samples']} | {resource['peak_rss']:,} | {resource['minimum_available']:,} | {value['maximum_scalar_error']:.9g} |")
    lines+=['','Both phases satisfy the declared 180-second, 6-GiB sampled group-RSS and 2-GiB available-memory guards. '
      'All observed PID/creation-time identities are terminal. Sampling does not establish an absolute memory ceiling or observe every short-lived child.','',
      f"Bundle SHA-256: `{pin(payload/'bundle.json')['sha256']}`. Audit SHA-256: `{pin(base/'audit.json')['sha256']}`.",
      'Full arrays, disassembly, source, resource observations and receipts are retained under '
      '`artifacts/e5-layernorm-amd-proof-20260920`. The core SHA-256 remains '
      '`48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710`.','']
    report.write_text('\n'.join(lines),encoding='utf-8');write(base/'terminal-verification.json',terminal)
    files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
    write(base/'closed.json',dict(schema=1,passed=True,closed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=audit['scope'],files=files,
        report=dict(file=report.as_posix(),**pin(report))))
    print('Closed',len(files),'files;',pin(base/'closed.json'))

if __name__=='__main__':main()
