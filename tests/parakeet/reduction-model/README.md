# Complete Parakeet check for shorter partial sums

This isolated candidate adds a 256-term partial-sum helper behind both existing
packed row kernels. It admits reductions of at least 1,024 terms and complete
32-column panels only. It accumulates into initial C, supports the final odd
row, and leaves short reductions, column tails, cache admission and Data intact.
Main product source and the live AMD campaigns remain unchanged.

Before model execution, geometry checks cover nonzero C, row/reduction/column
boundaries, exact original fallbacks and hardware-disabled refusal/public
execution. Original captured projections must reproduce the preceding selected
arithmetic. A compiled-instruction check permits only two changed methods and
one added helper; all other Core/Data methods must match frozen production.

The original byte-identical native consumer then checks all 784 arrays and
3,090,494 values on complete English/French/JFK, limit, silence and repeat
trajectories, plus rejection/recovery contracts. The independent original
auditor retains every numerical failure at the unchanged `1e-4` bound. A
separate original public consumer runs the twenty-clip corpus once.

From the repository root, execute once into fresh artifacts:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-model/prepare.py
    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-model/run.py
    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-model/audit.py

Normal .NET10.0.12, CPU2 inherited before runtime startup. Build/probe jobs
require 4 GiB preflight availability, 2 GiB RSS and 600 seconds; native replay
10 GiB / 8 GiB / 1,200 seconds; public corpus 14 GiB / 12 GiB / 1,200 seconds.
Every job requires 1 GiB available RAM, 20 GiB free disk and at most 1 GiB output.
Preflight waits at most 900 seconds. Keep failures; no automatic retry or changed
tolerance. No speed, AMD or product qualification follows this experiment alone.
