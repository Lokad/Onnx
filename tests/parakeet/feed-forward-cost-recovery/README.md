# Correct the diagnostic observer without repeating the Core build

Original build owner 1006487 is terminal with code 1. SDK identification, Core
restore/build and Data restore passed. Data build failed with CS1673: a lambda
inside the readonly observation scope captures the instance field `start`.
No diagnostic inference ran. The original sources, state, logs and collection
remain unchanged in `artifacts/parakeet-feed-forward-cost-amd-20260925`.

Copy that field to a local immediately before the predicate, and capture the
local instead. Require an exact inverse to the original observer bytes. Keep
all other Data source/project bytes, Core source, arithmetic, consumer, inspector,
workload and limits unchanged. Reuse the successfully built diagnostic Core;
only the corrected Data restore/build and previously unstarted inventory run.

Run with `C:/Python313/python.exe -X utf8 -B` from the repository root:

    tests/parakeet/feed-forward-cost-recovery/run.py prepare
    tests/parakeet/feed-forward-cost-recovery/run.py stage
    tests/parakeet/feed-forward-cost-recovery/run.py launch
    tests/parakeet/feed-forward-cost-recovery/run.py observe
    tests/parakeet/feed-forward-cost-recovery/run.py collect
    tests/parakeet/feed-forward-cost-recovery/review.py

The original VM namespace receives only additional recovery inputs and outputs;
every original staged input must still match its frozen digest. Recovery has
its own source manifest, owner, state, logs and collection. Local receipts live
in `artifacts/parakeet-feed-forward-cost-observer-recovery-20260925`.

The joint review retains the original failed build, checks all original and
recovery resource samples, and applies the original compiled arithmetic/public
contract checker. Require all 3,254 original Core methods, unchanged arithmetic
after removing the allowed markers, and all 697 original Data methods except
the certified Execute scope. Preserve the two existing Core warnings; corrected
Data must introduce none. Reuse the original application consumer byte for byte.

Only a passing joint review writes the original transport's capture prerequisite.
Then use the original `feed-forward-cost-diagnostic/run.py launch capture`,
`observe capture`, `collect capture` and `audit.py`, without editing those frozen
tools. Keep audit stdout outside both evidence directories. The failed e5 release
controls remain binding: this is a diagnostic, not product promotion.
