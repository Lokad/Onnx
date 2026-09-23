# M34 parakeet regression on AMD

Use the current selected Core `208371f6` / Data `b9358370` and the qualified
M34 Core `521bae17` / Data `f3b9aa81`. Complete Pyannote qualification must be
closed before preparation. Both products and all consumers remain unchanged.
Existing pinned model assets on the VM are reused.

From the repository root use `C:/Python313/python.exe -X utf8 -B` with
`run.py prepare`, `run.py stage`, `run.py launch`, then `run.py observe`.
Only collect terminal owners, then run `audit.py`. Never relaunch, overwrite
failed evidence, or observe a closed campaign. Numerical checkers are copied
byte for byte from the qualified M22 campaign; its checker tests still apply.

Keep every original native tolerance, exact selected output comparison,
complete public result, input/ownership check and resource bound. All request
clocks remain diagnostic, with no performance score. CPU2 workers execute
sequentially and CPU0 monitoring records each worker. Parakeet optimization
remains second to Pyannote; Whisper is deferred.

Four jobs cover 784 arrays / 3,090,494 values and 20 public clips per role.
Keep 12 GiB preflight/RSS ceiling and 1,800 seconds per job.
