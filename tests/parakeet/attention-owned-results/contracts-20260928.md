# Attention preparation passes focused numerical and ownership contracts

The isolated candidate changes only ComputationalGraph.PrepareOwnedMatMulWeights.
It admits square [1024,1024] weights for the five exact attention families while
preserving the existing feed-forward name/shape rules and all sharing guards.
No multiplication kernel changes. The qualified root product remains unchanged.

All **212 focused contracts pass with no skips**: 93 in normal AVX-512 mode,
93 with AVX-512 disabled, and 26 with hardware intrinsics disabled. This includes
all existing feed-forward and generic MatMul contracts and 29 new attention
cases. The geometry fact exercises all 38 T/2T-1 combinations from the 19 corpus
lengths in both Speed and Memory modes, with repeated requests. Results remain
bit-exact, held outputs stay independent, and per-call packing scratch is zero.

The remaining new cases verify paired name/shape eligibility, seven sharing or
visibility exclusions, unchanged existing cache entries, idempotence, exact
logical bytes including exceptional values, held original arrays, destination
alias rejection, collection of replaced arrays, fallback execution options and
preparation refusal without FMA. Exceptional-value byte checks concern data
movement; this fixture does not introduce a new arithmetic NaN-payload contract.

Compiled review 2a1bf8e3 confirms one changed original Core method, no additions
or removals, all 3,287 other Core and 697 Data methods unchanged, and unchanged
implementation flags, public declarations and assembly attributes. Data remains
byte-identical. The four existing warning occurrences are unchanged. All six
build jobs and three test jobs completed successfully, and every owner is terminal.

Candidate Core: ee5218dbab0a970b0f20f273fc2a839bb9b96e9d5438791529eb11c731d66859.
Data: 1ba343fd8b00fd85bddb33c57aaebf4217431955c40bbe45576448467e59c99f.
Frozen source/build tools: a1c61f8c. Source receipt: ee998786. Contract closure:
a293403a5dc3a7bdabb7fdc448470feea2b54dbf139d98b7b3f9ed0eef39854f.
[Complete results and test names](contracts-20260928.json) retain the evidence.

The actual-model census, complete native/numerical/public corpus and independent
application comparison remain required before admission. The predicted saving
comes from the measured 0.817785 seconds of repeated attention packing per corpus;
**no candidate speedup has yet been measured**. BENCHMARK.md remains unchanged.
