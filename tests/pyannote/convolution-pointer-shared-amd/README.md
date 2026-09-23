# M28 shared regression on AMD

Use the current selected Core `208371f6` / Data `b9358370` and the qualified
M28 Core `e776cec2` / Data `0c55b650`. Complete Pyannote qualification must be
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

Four jobs cover 166 arrays / 5,000,814 values per role, including five e5
inputs of up to 512 tokens. Preserve the original shared/e5 fixture inventory.
