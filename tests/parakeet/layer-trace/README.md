# Parakeet encoder boundary diagnostic

This opt-in local experiment investigates the retained Windows duration-logit
failure on the English replay clip. It captures the stem and all 24 encoder
layer outputs using both retained feature arrays, then tests propagation through
one native decoder call with fixed incoming recurrent states. It uses the
qualified current managed DLL and ORT 1.29.0. No production source is rebuilt.

Run from the repository root with `C:/Python313/python.exe -X utf8 -B`, followed
by `tests/parakeet/layer-trace/prepare.py`, `run.py`, then `analyze.py` in separate
commands. The historical asset locations are explicit in `common.py`. Existing
output paths are refused. Preparation builds the small capture consumer with
the repository SDK and `--tl:off`.

Twelve encoder workers run sequentially on Windows CPU2: managed/native engine,
native/managed features, and original/trace/trace-repeat model. A final native
worker evaluates all twelve resulting encoder arrays in the same decoder state.
Inputs, weights, sources, native libraries, binaries, outputs and process births
are retained. Resource limits are declared in `common.py`; failures stop the
schedule and preserve evidence.

Interpretation requires bit-identical original/trace finals and repeats, original
controls matching their retained outputs, and unchanged managed optimized nodes.
The analyzer checks every boundary against the unchanged scaled-error limit
`abs(actual-reference)/max(1,abs(reference)) <= 1e-4`, independently rechecking
maxima and failure counts with scalar arithmetic. A trace that changes the
computation is reported as such, not used to attribute the original failure.

This selected clip is a numerical diagnostic. Layer differences combine inherited
and local error; they do not identify an incorrect operator by themselves. The
fixed-state decoder is not an independent complete transcription trajectory.
No latency, broad accuracy or full numerical qualification claim follows.
