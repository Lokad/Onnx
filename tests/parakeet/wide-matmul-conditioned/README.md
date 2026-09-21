# Conditioned wide encoder MatMul comparison

This successor reuses the qualified twelve actual operand fixtures and unchanged
arithmetic from [the first probe](../wide-matmul/results-20260921.md). It does not
repeat encoder inference. The first grid's identical fallback controls differ
by 4.8–9.2%, so its apparent gains are not accepted.

Four fresh workers use normal .NET 10.0.12 on CPU 2. Each retains the original
144 synthetic geometry/accumulation tests, 32-call and 0.25-second warmup per
role, and every real-fixture bit comparison. The measured schedule now uses
twelve orders covering every directed pair of distinct preceding/current roles
once, with reversed orders in alternate workers. Three executions of the exact
upcoming role condition its working set before each measured call. All warmup,
conditioning and measured intervals remain in the raw files.

The three M=51 cells run identical public fallback code for every combined role.
Before a gain is attributed, each cell's aggregate maximum/minimum must be at
most 1.03 and every individual worker's at most 1.10. Limits were fixed before
launch. No samples or workers may be removed.

A candidate can enter a subsequent product trial only in the declared territory
M>=64 and M%3!=0: mean of the six cell ratios <=0.95, every cell <=1.01, and every
worker/cell <=1.05, after the controls pass. The existing three-row route remains
unchanged; all M=225 observations remain visible. These are experiment admission
rules, not evidence of application or ORT parity. Full trajectories and complete
API comparisons are still required before promotion.

From the repository root, using a new artifact only:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/wide-matmul-conditioned/prepare.py
    C:/Python313/python.exe -X utf8 -B tests/parakeet/wide-matmul-conditioned/finish.py

The preparation checks the complete parent closure and terminal identities,
builds against frozen production Core, and verifies that kernels and capture
source are byte-identical. The driver owns four sequential workers and the
independent audit. Do not edit frozen tools or launch a duplicate. Worker limits
are 600 seconds, 2 GiB RSS, 4 GiB available memory at preflight, at least 1 GiB
available during execution and 20 GiB free disk, and 64 MiB output. Only its owned
worker can be terminated. The VM and production assemblies remain unchanged.
