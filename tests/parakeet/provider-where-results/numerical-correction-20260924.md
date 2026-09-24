# Provider Where numerical checker correction

The first qualification stopped in the selected release on `capture-0`, before
any candidate worker ran. The new checker expected one validation stage. Source
inspection establishes four: `Tensor.Where` records one and its three calls to
`BroadcastTo` each record another. The failure did not serialize the actual
count, and no completed numerical result exists.

Closure `9bc68920404a05fd330454a9dab65c092cb28f9e82875ea409900f1e0939b550`
preserves all 84 collected files and 28 resource samples. Three build jobs passed;
the selected numerical worker exited -6 on the assertion. This neither admits
candidate correctness nor establishes a candidate defect.

The distinct `provider-where-numerics-v2` tools preserve products, all 203 cases,
20 provider contracts, both instruction modes, bit oracles and resource bounds.
They record and check complete ordered stages: four validations for ordinary
Where, one for successful specialization, five for attempted specialization
followed by the original fallback. The earlier namespace is closed unchanged.
