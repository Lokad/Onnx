# Qualified DINOv3 output hash pairs

The constant-input DINOv3 test preserves two complete frozen output hash pairs.
They were independently checked against every native output value; they are not
selected from the current output or regenerated during testing.

| Observed configuration | last_hidden_state | pooler_output |
|---|---:|---:|
| Windows, AVX2, .NET 10.0.12 | 2137279831586830888 | 6070936888732424120 |
| Linux AMD EPYC 9V74, AVX-512, .NET 10.0.8 | 12756423648221837382 | 6147948217399512682 |

These labels describe the observed environments; they do not establish which
platform difference causes the rounding difference. The test requires one
complete pair, retains its native-reference mean checks, and does not change
numerical tolerances or skip the AMD result.

During default qualification at `a6b025d`, 2,881 AMD non-CLI tests passed and
the single-platform DINO hash assertion failed. Eleven isolated diagnostic
processes reproduced the same AMD hash: default, all nine selected mechanisms
disabled, and each mechanism disabled individually. None restored the Windows
hash. The failed suite and every diagnostic remain archived.

Fresh default and all-nine-disabled replays each pass all 106 native
DINOv3/ResNet50/GPT-2 output arrays, totaling 1,286,766 values. Every array is
also byte-identical to the preserved AMD default replay using the earlier core
`ebfb87220c4f980f21d7f39b2152c651104173bad3f5cfd3c632aecc4fb7bd3a`.
That earlier replay already contains the AMD hash pair above. The maximum
full-array DINO scaled error is `1.98036432266e-5`, below the unchanged `1e-4`
gate. The original Windows pair remains unchanged and passes locally.

Evidence is retained locally under `artifacts/production-defaults-hash-20260919`:
`audit.json` records both complete comparisons and all eleven diagnostics.
The result archive SHA-256 is
`9578dfd74388d3bcc80c90544ac5a406cbf420d7d2c38504a863f8f7f609248d`.
The initial failed suite is preserved under `artifacts/production-defaults-20260919`,
archive SHA-256
`1ddde56d92aeafe2e3e9586b5fbb37730e242d3ead312762c045ff6b78e98d6b`.
