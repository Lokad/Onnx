# Correct the reversed-stride test expectation

The first suite passed394 of395 tests. Its new reversed-stride assertion
incorrectly treated the backing buffer's order as enumeration order. For a
[2,3] reversed-stride parent the strides are [1,2]; selecting columns1 and2
gives logical rows [2,4] and [3,5]. Enumeration is [2,4,3,5], while independently
owned reversed-stride storage is [2,3,4,5]. Assert both separately. Empty-slice
and reversed-stride preservation checks stay in that same case.

The product helper rejects reversed layouts, so the reviewed unchanged base
conversion handles this case. No product change or rebuild is needed. Compile
only the corrected test assembly against exact inspected Core49c3a958. Preserve
the failed suite and its one failure. Run all395 cases in both instruction
modes after the corrected test build passes. The existing numerical, ownership,
identity, source-policy and resource gates remain unchanged.

Run prepare/stage/launch build, observe and collect only that terminal owner,
review_build.py, then launch capture, observe, collect and audit.py. Use local
Python3.13 -X utf8 -B, exclusive AMD CPU2 with CPU0 monitoring, .NET10.0.204 /
runtime10.0.8 and --tl:off. Every namespace is single-use and frozen on prepare.
