# Complete-request attribution for the selected Parakeet release

Use unchanged Core `672e5f30` / Data `065b7a7f` and consumer `a196f652`
(49,664 bytes). Build closure `b5364b30` verifies all 162 consumer methods:
161 unchanged, Main differs at only its two product-hash strings, every
implementation flag and public interface exact. Both complete-public-call
markers retain NoInlining alone. No product rebuild or managed overrides.

Run the fixed twenty-clip / 213.265-second corpus in control, sampled-a,
sampled-b order. Each fresh process completes one warmup and three measured
passes: 240 public calls, 180 measured. Preserve every clock, original native
and public check, exact selected result, input hash and owned-output check.
The FullParakeet marker covers complete transcription including frontend,
encoder/decoder inference, greedy decoding and owned results.

Capture only after all twenty warmups. Reconcile both Speedscope and Chromium
exports completely. Every selected stack must belong to the original target
thread; warmup markers must be absent. Each capture's complete-corpus sampled
duration must agree with summed measured wall clocks within the original 5%
bound. Publish all exclusive/inclusive weights and separate wall/process CPU
clocks and collection overhead. No per-clip attribution is inferred from merged
sample intervals. These diagnostics cannot establish a new ORT ratio or gain.

Fixed limits: target CPU2, collector/monitor CPU0; 12 GiB available before each
target and combined owned RSS cap, 900 seconds/pair, 1 GiB remaining memory/tmpfs
and output/pair, 2 GiB total artifacts. Initial tmpfs 3 GiB. Exporters retain
8 GiB available/RSS, 900 seconds and the existing output/disk bounds. Models,
offline runtime and tracer are reused from their pinned locations.

From the repository root run `C:/Python313/python.exe -X utf8 -B` with
`prepare.py`, then `transport.py stage`, `launch`, `observe`, `collect` separately.
After terminal capture owners, run `export.py launch`, `observe`, `collect`,
then `audit.py`. Refuse existing output; resume only the same verified PID/birth.
No tuning, trimming, favorable retry, maintenance during workers or model copies.

The rejected M61 control `e9c296b8` parks M59. The present plan refreshes attribution
before selecting another hypothesis. A possible inclusive 4096 packing boundary
requires a separate plan and qualification at unchanged 256/64 MiB budgets;
its old Windows observations do not establish current AMD performance.
