# Shared-model and suite qualification for corrected partial sums

These tools qualify the isolated Core produced by `../reduction-dispatch`.
They preserve original consumer binaries and native fixtures. They do not
change the main product source or the frozen AMD campaigns.

From the repository root, the successful shared-model sequence is:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-shared/qualify_v2.py prepare
    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-shared/qualify_v2.py run
    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-shared/qualify_v2.py audit
    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-shared/qualify_suites.py

`qualify.py` retains the failed initial preparation: it incorrectly required
Data.dll in the original Core-only replay runtime. No inference started.
The separate v2 preserves the consumer's original dependencies and checks all
166 native e5/DinoV3/ResNet50/GPT-2 arrays. Its independent audit qualifies the
complete DinoV3 hash pair before the suite copy may accept that additional pair.
Existing hashes and numerical bounds remain unchanged.

The subsequent pyannote sequence is:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-shared/qualify_pyannote.py prepare
    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-shared/qualify_pyannote.py run
    C:/Python313/python.exe -X utf8 -B tests/parakeet/reduction-shared/qualify_pyannote.py audit

It reuses the original graph/application consumer for three ten-second crops,
18 graph arrays and sixteen public dialogue requests. All values, speaker
timelines, centroids and held-output checks remain part of qualification.

Model/suite workers use CPU2, 10 GiB available-memory preflight, 8 GiB process
RSS and 900 seconds. Builds use 4 / 2 GiB. All retain 1 GiB available RAM,
20 GiB free disk and 1 GiB output guards. Never rerun a closed experiment into
the same artifact path; preserve failures and use explicit additive successors.
These checks supply no matched latency result or AMD/product promotion.
