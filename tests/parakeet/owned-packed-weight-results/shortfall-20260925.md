# Where the owned-weight application gain falls short

The unchanged complete-application admission fails: 2.590% gain against the
required 3%, despite all 63 repeatability controls and all 20 per-clip gates
passing. The failed verdict remains. This review partitions every clip using
the reconstruction condition established before timing; it creates no new score
or admission rule.

| Existing execution path | Clips | Selected seconds | Candidate seconds | Saved seconds | Latency change |
|---|---:|---:|---:|---:|---:|
| No weight reconstruction | 13 | 38.830513 | 36.955514 | 1.874998 | 4.829% lower |
| Dense weights reconstructed for final row | 7 | 20.530310 | 20.867992 | -0.337681 | 1.645% higher |

All 13 clips without reconstruction improve. Six of the seven reconstruction
clips become slower; the seventh improves by only 0.247%. Those seven lengths
are 167, 89, 157, 83, 61, 151 and 169: odd values not divisible by three.
The actual counter audit recorded 609 additional 16 MiB reconstructions,
10,217,324,544 cumulative copy bytes, on exactly those inputs in both modes.

This alignment identifies a specific diagnostic target, not a causal timing
attribution. The groups contain different sequence lengths and original costs.
The 0.338-second group regression is not the cost of copying: it is the net
effect of saved packing, reconstruction and any cache/runtime changes.
Do not add the 1.875-second saving from the other group to an assumed copy cost.

The next bounded check is to retain the existing CopyY stage durations and
complete feed-forward call durations on these exact products and clips. The
existing counter consumer already observes those stages but discards their
times. Reuse the unchanged Core/Data binaries, preserve every clip and result,
and check repeatability before choosing a remedy. No cache-size sweep, new
arithmetic kernel or unchanged application retry is selected.

[Complete application result](application-20260925.md),
[prior actual packing/reconstruction counts](counters-20260925.md),
[every clip](shortfall-20260925.csv), [identities and group totals](shortfall-20260925.json).
The qualified product and BENCHMARK.md remain unchanged. No new model execution
was needed for this review.
