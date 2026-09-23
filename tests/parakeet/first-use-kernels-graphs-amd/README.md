# M43 shared graph performance regression and ORT comparison

Run only after full Parakeet admission and fresh shared/e5 and Pyannote output
qualification. Reuse the exact release graph consumer and native runner, with
fresh processes for five e5 cases, DINOv3, ResNet50 and GPT-2. Core identity is
provided by a per-product manifest. The unchanged managed consumer labels both
products `current` internally; job identity and actual Core hashes distinguish
them throughout collection and analysis.

First run all 24 numerical processes (three calls each), then six timing
processes per case in current, candidate, ORT, ORT, candidate, current order.
Each timing process retains 60 warmups and 60 measurements. All 5,832 clocks,
2,880 measurements and 72 setup intervals are retained. Never pool campaigns,
trim clocks or repeat unchanged failures. Both roles must stay within the
original native 1e-4 bounds; candidate and current output bytes must match.

Admission requires every engine/case process-mean max/min ratio at most 1.10
(24 controls) and candidate latency at most 1.05 of current on every case
(eight regression gates). Compute process means and ratios using exact integer
clock fractions. Publish candidate/ORT ratios only after complete product and
actual-root qualification; this lane does not establish transcription parity.

Use `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`, `launch`,
`observe`, `collect`, then `audit.py`. No consumer rebuild or model download is
needed. CPU 2 computes, CPU 0 monitors; preflight 12 GiB available/3 GiB tmpfs,
worker 8 GiB RSS/900 seconds, campaign four hours. Freeze before execution and
retain any failure without changing the gates.
