# Whisper real-speech numerical localization — September 19, 2026

The full managed Whisper tensor gate still fails on this fixed twenty-recording corpus. Giving the same managed decoder the native encoder output makes all 707 complete logit arrays pass, including the repeated first recording. This localizes the observed logit failures to differences in the encoder states supplied to decoding. It does not distinguish frontend perturbations from encoder arithmetic, prove that a particular kernel is wrong, or qualify the full pipeline through the diagnostic path.

Both paths independently advance their own caches and chosen tokens. Every transcript, token sequence, stop and no-speech decision matches the native reference. Their first recording also repeats with identical output bits. The existing [labeled accuracy result](../audio/accuracy/results-20260919.md) remains 10 word errors in 559 reference words; this investigation adds no recordings or new WER claim.

| Compared stage | Arrays | Values | Failed arrays | Failed values | Maximum scaled error |
|---|---:|---:|---:|---:|---:|
| Managed features | 21 | 8,064,000 | 0 | 0 | 1.1920929e-07 |
| Managed encoder output | 21 | 40,320,000 | 21 | 323,293 | 0.004583925 |
| Decoder from managed encoder | 707 | 39,936,820 | 405 | 1,245,368 | 0.00057518482 |
| Decoder from native encoder (diagnostic) | 707 | 39,936,820 | 0 | 0 | 6.1839819e-05 |

Counts include the twenty recordings and one repeat. Hidden/logit comparisons retain `abs(actual-reference) / max(1, abs(reference)) <= 1e-4`; all arrays must be finite and shapes exact. Features also pass their separate absolute `1e-5` requirement: maximum absolute error is 1.1920929e-07. The raw feature ledger additionally records scaled errors. No gate was loosened. Across both paths, the independent audit recomputes 128,257,640 values in 1,456 full saved tensors.

The largest relative L2 error—Euclidean error divided by reference-vector length—is 0.00011100352 for managed encoder outputs, 0.00017846871 for their decoder logits, and 1.297387e-05 for decoder logits given native hidden states. These describe the error; they are not substitute acceptance thresholds.

| Recording | Encoder maximum scaled error | Logits from managed encoder | Logits from native encoder |
|---|---:|---:|---:|
| 121-121726-0000 | 0.00296792 | 0.00037393 | 3.51667e-05 |
| 121-123852-0000 | 0.00187063 | 0.0001598 | 3.32594e-05 |
| 237-126133-0002 | 0.000837088 | 0.000575185 | 3.80278e-05 |
| 237-126133-0000 | 0.00238617 | 0.00037694 | 3.24249e-05 |
| 260-123286-0000 | 0.00201717 | 0.000379324 | 3.21269e-05 |
| 260-123288-0005 | 0.00185955 | 0.000190794 | 5.06192e-05 |
| 672-122797-0000 | 0.00172263 | 0.000220001 | 2.65837e-05 |
| 672-122797-0001 | 0.00283977 | 0.000237375 | 3.06368e-05 |
| 908-157963-0005 | 0.00183874 | 0.000308126 | 3.61126e-05 |
| 908-157963-0000 | 0.00363666 | 0.000276923 | 3.90112e-05 |
| 1089-134686-0002 | 0.00458393 | 0.000365138 | 3.36766e-05 |
| 1089-134686-0011 | 0.00110149 | 0.000421882 | 2.46763e-05 |
| 1188-133604-0001 | 0.00200465 | 0.000348076 | 3.15309e-05 |
| 1188-133604-0002 | 0.00153371 | 0.0002639 | 3.74913e-05 |
| 1221-135766-0002 | 0.00391188 | 0.000380695 | 6.18398e-05 |
| 1221-135766-0000 | 0.00114077 | 0.000232816 | 2.98619e-05 |
| 1284-1180-0000 | 0.00144547 | 0.000195561 | 3.63588e-05 |
| 1284-1180-0018 | 0.00205362 | 0.000226051 | 3.09348e-05 |
| 1320-122612-0001 | 0.0021607 | 0.000284255 | 3.50475e-05 |
| 1320-122612-0000 | 0.0015766 | 0.000335336 | 3.19481e-05 |

## Effect on the recorded token decisions

An additional audit examines every full logit vector after applying the same suppression policy. If the native winning token exceeds the runner-up by more than twice the maximum absolute allowed-logit difference, that error bound suffices to preserve the winner. All 707 decisions satisfy this condition on the fully managed path, including EOS and the repeated recording. The smallest margin divided by twice the observed maximum difference is 2.956731; the smallest positive margin left after that allowance is 0.000776290894. The diagnostic path also satisfies it on all 707 decisions.

This is a bound using the observed arrays on matching token histories. It does not establish a bound for unseen speech, perturbations of the audio, other languages or longer recordings. The strict full-tensor failures above remain failures.

## Execution and evidence

The corpus is the previously fixed ten-speaker, twenty-recording clean-English subset: 213.265 seconds and 559 labeled words. Its native ORT 1.29.0 reference already saved complete encoder states and logits. Every reused file is verified against the prior corpus receipt; no native reference was regenerated or substituted after seeing new results.

The artifact replay loads the exact qualified c6bf781 core and Data binaries. Through artifact-only reflection it uses WhisperTranscriber's three prepared graphs and the actual WhisperGeneration.Decode method, with the same Memory contexts and 256/64/64-MiB packed budgets. A callback observes graph calls and saves full arrays. The fully managed path computes features from PCM and its own encoder state. The separately named native-hidden diagnostic replaces only that initial state; each decoder then owns its cache/token progression. The public API and production code are unchanged.

Windows .NET 10.0.12, CPU 2, normal runtime and no forced GC were used. All actual feeds remain unchanged, and 36 first-request output tensors are held through later calls and resets. The audit verifies each later cache against the matching earlier managed output hash and derives token choices/confidence directly from saved logits. Native ORT is absent from the managed process. Eight corrupted-evidence checks include rejecting diagnostic success as end-to-end success.

Worker 39176 is terminal with exit 2, explicitly representing the retained numerical failure after completing all 42 composed path requests. Maximum sampled process-tree RSS is 10,510,155,776 bytes; process-reported peak is 10,517,520,384 bytes, below the 16-GiB guard. Output capture and held tensors change lifetimes, so these are diagnostic resource observations, not a new public-API memory bound or performance benchmark. The VM was not used.

An initial compile-time explicit-interface mistake and a preparation receipt-key mistake were corrected before inference. The failed preparation source is preserved; the existing manifest and all dependencies were independently reverified before freezing. Source and evidence remain under `artifacts/whisper-numerical-20260919`. Its completed prepare, supervisor, audit and receipt writers are single-use.

Core SHA256: `7653c1686419d612e2624740908a44ffdee91b4239628a979bac44f5b6b863e9`. Data SHA256: `13899fea53b970594631dae62d43eaaee4695b3f1ff096bbb5dd059e0d21760f`. Frozen payload receipt SHA256: `3f8746f6cfc505288b3c2aac23da101f39729b90c334e2fa41ebdafe08e5a9ea`. Independent numerical receipt SHA256: `519cd75cb6383056d10d2be015dce92880b18a308d532f5bf5c7043db7ad7899`. Decision-margin evidence SHA256: `fd7638edfaee754e676d6d7e319646c6bcf328dcc26a811a3276b55cdb78d506`.

Further numerical work should isolate encoder input perturbations and encoder arithmetic against an independent higher-precision reference. These results do not support tuning decoder arithmetic to compensate for the encoder, changing the existing tolerance, or declaring broader multilingual/noisy/long-recording support complete.
