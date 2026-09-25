# Identify the two predicted ORT depthwise routes

This diagnostic reuses the exact installed ORT library and the original public
Parakeet instruction samples. It runs no inference and changes no product binary.
The revision is `2e2543fbe9fae542f921d47a72d21d5a4ef0b710`.

`run.py` assembled the convolution object, then stopped at an unsupported internal
relocation. Its failed closure is retained. `finish.py` reuses that exact object,
resolves all 52 internal calls, and assembles the previously unstarted M1 object.
All 53,162 convolution text bytes match the installed library uniquely after
resolution. All 652 M1 bytes match after resolving its two constant references;
the referenced eight integers match the exact source table. No relocation bytes
are merely ignored. ELF executable mappings are checked explicitly.

Both function intervals contain measured-request instruction addresses from the
original capture. Shared convolution epilogues outside the depthwise entry/local
helper interval are excluded. Samples establish execution in the public workload;
they do not establish invocation counts or exclusive attribution to encoder nodes.

The first three commands have already completed; do not repeat them:

    C:/Python313/python.exe -X utf8 -B -m unittest discover -s tests/parakeet/depthwise-native-proof -v
    C:/Python313/python.exe -X utf8 -B tests/parakeet/depthwise-native-proof/run.py
    C:/Python313/python.exe -X utf8 -B tests/parakeet/depthwise-native-proof/finish.py

Five tests cover changed instruction bytes, wrong constants, ambiguous or
non-executable matches, nonzero ELF load addresses and exact internal resolution.
Evidence is under `artifacts/parakeet-ort-depthwise-kernels-20260925`.
