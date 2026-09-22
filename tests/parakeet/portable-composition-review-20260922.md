# Parakeet arithmetic after portable pyannote integration

The normal portable pyannote candidate leaves `TensorOps.MatMul.cs` identical,
after BOM/newline normalization, to the `b0f3ff10` preimage used by the qualified
Parakeet arithmetic dispatcher. Thus the existing two-method dispatcher change
can be composed without resolving an overlapping source edit. This is a source
inspection; it does not qualify a new binary or establish a speedup.

The retained [arithmetic qualification](reduction-dispatch/results-20260921.md)
clears all three Windows native discrepancies at the unchanged `1e-4` bound:
784 arrays / 3,090,494 values pass. It changes dynamic and prepared dispatch,
plus one internal helper that uses 256-term partial sums. All original raw
matrix kernels remain unchanged. Existing AVX-512 precedence, packing budgets,
mutable-weight checks and fallback paths remain intact. The same dispatch
admission must be retained; do not narrow it to avoid failing tests.

Its [Windows timing trial](arithmetic-comparison/results-20260921.md) failed
repeatability and speed-selection gates. That failure remains evidence, and
this review does not select it as a performance improvement. The numerical
fix can be evaluated as a distinct composition on the selected normal pyannote
source, followed by complete qualification and its first target-VM comparison.
Do not repeat the unchanged failed Windows timing protocol.

After banking pyannote, overlay only the two known dispatcher methods and the
qualified helper in an isolated normal source build. Require exact compiled
identity for every other Core method and all Data methods, unchanged public
APIs, preserved raw-kernel bits, the existing geometry/hardware-off probes,
complete native Parakeet trajectories and all twenty public clips. Requalify
e5, DinoV3, GPT-2, ResNet50 and pyannote because the arithmetic is shared. Any
DinoV3 hash accepted by the test suite must first be justified against complete
native arrays; retain previous accepted pairs and existing numerical bounds.
Then run normal complete suites and measure complete applications against the
new production baseline and fresh ORT under a prospectively fixed protocol.

Packing eligibility at reduction width 4096 is a separate change from the
arithmetic. Preserve the existing 256 MiB encoder budget until actual memory and
complete application timing justify another budget. The prior 2032 MiB trial
failed its available-memory guard before an ORT worker started; it provides
no speed comparison or selected budget. New performance work should keep the
[measured encoder MatMul share](performance-profile/results-20260921.md) as a
prioritization clue, without claiming it measures the current AMD composition.

Pyannote retains the active VM. This review launches no process, edits no
product, changes no default and adds no new performance result. Whisper stays
deferred.

Exact source files inspected:

| Source | Bytes | SHA256 |
|---|---:|---|
| portable MatMul | 46624 | `703dabfe07bd85e93fed3090b4255c196459f6cd26f5b01c000be4105ff368ae` |
| qualified arithmetic MatMul | 46950 | `f0a17a16848324b5be3b465854103aa3d539df228c93896e2b2afac01dcafbc9` |
| qualified arithmetic helper | 2788 | `94617be5d0ad04ac1c1782eedfa55a45200e5c74470bb5be68105dc3037a99c9` |
