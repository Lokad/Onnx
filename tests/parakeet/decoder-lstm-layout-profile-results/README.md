# Combine fresh profiles before choosing another optimization

The publisher is pending the selected LSTM layout's actual release qualification
and both fresh profile captures. There is no current partition or new timing
result here yet. Keep the existing application verdict and BENCHMARK.md separate.

Reuse the exact reviewed encoder memberships from mapping closure 17f0e968,
which binds analysis 0faf7f22. These are distinct files with distinct digests.
Keep all 2,856 managed and 1,993 native encoder nodes exactly once, the other
graph phases, time outside nodes and time outside graph calls. Before reuse,
require identical managed descriptors and native node metadata, runtime shapes,
node/session call counts and complete coverage across all three graphs.

Every timing comes from the new managed and native captures. Historical reports
supply membership and descriptors only. Missing or overlapping groups, changed
shapes, altered calls or duplicate descriptors refuse the comparison. Controls
and profiling overhead remain separate, with no subtraction. The partition
identifies work to investigate; it neither proves an internal kernel route nor
chooses a candidate. Inspect the exact applicable ORT source or trace next.

The latest retained managed partition is the post-padding profile in
../pad-current-profile-results/diagnosis-20260927.md: 50.142916 seconds, with
September 24 ORT node clocks. It predates sigmoid and both decoder changes.
The earlier 53.594489-second pre-padding partition is not the latest one.
Neither supplies timing to the new publisher.

The local tests use real retained captures to reproduce every historical group,
then prove that poisoned historical clocks do not change the newly supplied
clocks. They also reject missing/overlapping memberships and native shape,
call-count, node-coverage and duplicate-descriptor changes. They perform no
inference and are not evidence of a future profile or product result.

From repository root, prefix with C:/Python313/python.exe -X utf8 -B:

    -m unittest discover -s tests/parakeet/decoder-lstm-layout-profile-results -p test_*.py

After the actual root qualifies and both profile audits pass, publish once:

    tests/parakeet/decoder-lstm-layout-profile-results/publish.py

The publisher verifies both captures and their shared root/application identity,
then writes fresh evidence under artifacts/parakeet-decoder-lstm-layout-gap-20260927
and observations-20260927.json / diagnosis-20260927.md here. Preserve failed
captures and original clocks; do not replay a completed process or replace a
failed profile with historical timings.
