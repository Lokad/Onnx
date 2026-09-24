# Isolated prepared decoder recurrence

This source candidate follows the actual two-node capture, closure `28c7afe4`.
It adapts the reviewed voice-branch preparation idea to current graph lifecycle,
source identity and the existing aggregate packing budget. It reuses the exact
current `LstmProjectOrdered` method and leaves its gate arithmetic unchanged.

`prepare.py` verifies all 422 selected inputs, the complete capture closure and
the prior source review. It creates an isolated 424-file snapshot under
`artifacts/parakeet-prepared-recurrence-source-20260924`, with six product files
changed/added and one focused test file. It never edits root product source.
Run from the repository root with `C:/Python313/python.exe -X utf8 -B`.

The new internal map owns constant W/R transposes separately from mutable graph
bindings. Each pair is admitted only if its missing arrays fit the capacity left
after current matrix/convolution preparation. Refresh first prunes convolution,
then complete recurrent pairs using that already-retained byte total, then
reconciles matrix storage. Invalidation clears all maps. Execution contexts and
provider options share the map; source reference, array, shape and layout guards
refuse stale operands. Alias wrappers cannot overwrite an existing mapping.
Encoder/decoder caps stay 256/64 MiB.

Only a forward, one-direction, one-step, batch-one, input/hidden-size-640 call with
enabled hardware SIMD can use both prepared weights. Unsupported geometry,
scalar mode, direct calls and missing/stale maps keep the original routes.

Focused contracts cover atomic zero/short/exact budgets, repeated preparation,
all-map accounting, held outputs, immutable sources, replacement, shape and
consumer changes, aliases, offset/reversed views, invalidation/rebuild and direct/
scalar/unsupported-geometry fallbacks. An internal test poisons prepared arrays
after ordinary agreement: fresh graph contexts must then expose the poison,
while scalar/direct calls remain exact, proving execution-context routing.

This snapshot is not a selected release. AMD-only build and compiled scope,
both-mode contracts, actual-model residency/dispatch, full captured/native/public
numerics, prospectively fixed complete-call timing and full application admission
are still required. Every unrelated method body/flag and public interface must
remain exact. Promotion additionally requires shared/e5/Pyannote, root/suites and
package qualification. Preserve every failure and use named successors for
harness fixes; do not change a frozen source candidate in place.
